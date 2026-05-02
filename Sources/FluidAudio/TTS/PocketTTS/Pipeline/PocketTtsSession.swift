@preconcurrency import CoreML
import Foundation

/// A persistent TTS session that keeps the voice KV cache warm across utterances.
///
/// Creating a session performs the expensive voice prefill once (~125 tokens),
/// then each enqueued utterance only pays the text prefill cost. Mimi decoder
/// state persists across utterances for seamless audio continuity.
public actor PocketTtsSession {

    private static let logger = AppLogger(category: "PocketTtsSession")

    // MARK: - Public Interface

    /// Stream of generated audio frames (80ms / 1920 samples at 24kHz each).
    ///
    /// Frames are yielded as soon as they are generated. The stream completes
    /// after `finish()` is called and all enqueued text has been synthesized,
    /// or immediately if `cancel()` is called.
    public nonisolated let frames: AsyncThrowingStream<PocketTtsSynthesizer.AudioFrame, Error>

    /// Enqueue text for synthesis.
    ///
    /// Non-async and safe to call from any isolation context. Text is chunked
    /// internally if it exceeds the per-chunk token limit. Can be called
    /// multiple times to stream text as it arrives.
    public nonisolated func enqueue(_ text: String) {
        textContinuation.yield(text)
    }

    /// Signal that no more text will be enqueued.
    ///
    /// The session will finish generating all previously enqueued text,
    /// then complete the `frames` stream.
    public nonisolated func finish() {
        textContinuation.finish()
    }

    /// Cancel ongoing generation and finish the frames stream.
    ///
    /// Awaits until the generation task has fully stopped — after this returns,
    /// no more CoreML predictions are running and the Neural Engine is free.
    public func cancel() async {
        generationTask?.cancel()
        textContinuation.finish()
        await generationTask?.value
    }

    // MARK: - Internal State

    private nonisolated let textContinuation: AsyncStream<String>.Continuation
    private let textStream: AsyncStream<String>
    private let frameContinuation: AsyncThrowingStream<PocketTtsSynthesizer.AudioFrame, Error>.Continuation
    private var generationTask: Task<Void, Never>?

    // Models
    private let condModel: MLModel
    private let stepModel: MLModel
    private let flowModel: MLModel
    private let mimiModel: MLModel
    private let condLayerKeys: PocketTtsLayerKeys
    private let flowlmLayerKeys: PocketTtsLayerKeys
    private let mimiKeys: PocketTtsMimiKeys

    // Persistent state
    private let voiceKVSnapshot: PocketTtsSynthesizer.KVCacheState
    private let constants: PocketTtsConstantsBundle
    private let bosEmb: MLMultiArray
    private let temperature: Float
    private var mimiState: PocketTtsSynthesizer.MimiState
    private let flowScratch: PocketTtsSynthesizer.FlowDecoderScratch
    private var rng: SeededRNG

    // MARK: - Initialization

    /// Create a session with pre-computed voice KV cache.
    ///
    /// This initializer is internal — use `PocketTtsManager.makeSession()` instead.
    init(
        voiceKVSnapshot: PocketTtsSynthesizer.KVCacheState,
        mimiState: PocketTtsSynthesizer.MimiState,
        constants: PocketTtsConstantsBundle,
        condModel: MLModel,
        stepModel: MLModel,
        flowModel: MLModel,
        mimiModel: MLModel,
        condLayerKeys: PocketTtsLayerKeys,
        flowlmLayerKeys: PocketTtsLayerKeys,
        mimiKeys: PocketTtsMimiKeys,
        bosEmb: MLMultiArray,
        temperature: Float,
        seed: UInt64
    ) throws {
        self.voiceKVSnapshot = voiceKVSnapshot
        self.mimiState = mimiState
        self.constants = constants
        self.condModel = condModel
        self.stepModel = stepModel
        self.flowModel = flowModel
        self.mimiModel = mimiModel
        self.condLayerKeys = condLayerKeys
        self.flowlmLayerKeys = flowlmLayerKeys
        self.mimiKeys = mimiKeys
        self.bosEmb = bosEmb
        self.temperature = temperature
        self.flowScratch = try PocketTtsSynthesizer.FlowDecoderScratch(model: flowModel)
        self.rng = SeededRNG(seed: seed)

        // Text queue channel
        let (textStream, textContinuation) = AsyncStream.makeStream(of: String.self)
        self.textStream = textStream
        self.textContinuation = textContinuation

        // Frame output stream
        let (frames, frameContinuation) = AsyncThrowingStream.makeStream(
            of: PocketTtsSynthesizer.AudioFrame.self
        )
        self.frames = frames
        self.frameContinuation = frameContinuation
    }

    /// Start the generation loop. Must be called once after init.
    func start() {
        generationTask = Task { [weak self] in
            guard let self else { return }
            await self.generateLoop()
        }
        frameContinuation.onTermination = { [weak self] _ in
            guard let self else { return }
            Task { await self.cancel() }
        }
    }

    // MARK: - Generation Loop

    private func generateLoop() async {
        var utteranceIndex = 0
        var isFirstFrame = true
        let sessionStartTime = CFAbsoluteTimeGetCurrent()

        do {
            for await text in textStream {
                if Task.isCancelled { break }

                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { continue }

                let enqueueTime = CFAbsoluteTimeGetCurrent()
                let chunks = PocketTtsSynthesizer.chunkText(
                    trimmed, tokenizer: constants.tokenizer
                )
                if PocketTtsConstants.detailedTimingLogsEnabled {
                    Self.logger.info(
                        "Session enqueued '\(trimmed)', \(chunks.count) chunk(s)")
                }

                for (chunkIndex, chunkText) in chunks.enumerated() {
                    if Task.isCancelled { break }

                    try await generateChunk(
                        text: chunkText,
                        chunkIndex: chunkIndex,
                        chunkCount: chunks.count,
                        utteranceIndex: utteranceIndex,
                        isFirstFrame: &isFirstFrame,
                        enqueueTime: enqueueTime,
                        sessionStartTime: sessionStartTime
                    )
                }
                utteranceIndex += 1
            }
            frameContinuation.finish()
        } catch {
            frameContinuation.finish(throwing: error)
        }
    }

    private func generateChunk(
        text: String,
        chunkIndex: Int,
        chunkCount: Int,
        utteranceIndex: Int,
        isFirstFrame: inout Bool,
        enqueueTime: CFAbsoluteTime,
        sessionStartTime: CFAbsoluteTime
    ) async throws {
        let chunkStart = CFAbsoluteTimeGetCurrent()
        let (normalizedChunk, framesAfterEos) = PocketTtsSynthesizer.normalizeText(text)
        if PocketTtsConstants.detailedTimingLogsEnabled {
            Self.logger.info("Session chunk \(chunkIndex): '\(normalizedChunk)'")
        }

        // Tokenize; prefill reads embeddings directly from the shared table.
        let tokenIds = constants.tokenizer.encode(normalizedChunk)

        // Clone voice KV snapshot and prefill text tokens only
        let cloneStart = CFAbsoluteTimeGetCurrent()
        var kvState = try PocketTtsSynthesizer.cloneKVCacheState(voiceKVSnapshot)
        let cloneMs = (CFAbsoluteTimeGetCurrent() - cloneStart) * 1000

        let prefillStart = CFAbsoluteTimeGetCurrent()
        kvState = try await PocketTtsSynthesizer.prefillKVCacheText(
            state: kvState, tokenIds: tokenIds, constants: constants, model: condModel,
            layerKeys: condLayerKeys
        )
        let prefillMs = (CFAbsoluteTimeGetCurrent() - prefillStart) * 1000
        if PocketTtsConstants.timingLogsEnabled {
            Self.logger.info(
                "[Timing] chunk=\(chunkIndex) kvClone=\(String(format: "%.1f", cloneMs))ms textPrefill=\(String(format: "%.1f", prefillMs))ms tokens=\(tokenIds.count)"
            )
        }

        // Generation loop
        let maxGenLen = PocketTtsSynthesizer.estimateMaxFrames(text: text)
        var eosStep: Int?
        let sequence = try PocketTtsSynthesizer.SequenceScratch()
        sequence.writeNaN()
        let flowLMInputProvider = PocketTtsSynthesizer.FlowLMInputProvider(
            sequence: sequence.array,
            bosEmb: bosEmb,
            layerKeys: flowlmLayerKeys
        )
        let totalFramesAfterEos = framesAfterEos + PocketTtsConstants.extraFramesAfterDetection

        // Accumulators for per-frame timing summary
        var flowLMTotalMs: Double = 0
        var flowDecodeTotalMs: Double = 0
        var mimiTotalMs: Double = 0
        var frameCount = 0

        for step in 0..<maxGenLen {
            if Task.isCancelled { break }

            let frameStart = CFAbsoluteTimeGetCurrent()

            // FlowLM step
            let flowLMStart = CFAbsoluteTimeGetCurrent()
            let (transformerOut, eosLogit) = try await PocketTtsSynthesizer.runFlowLMStep(
                state: &kvState,
                model: stepModel,
                layerKeys: flowlmLayerKeys,
                inputProvider: flowLMInputProvider
            )
            let flowLMMs = (CFAbsoluteTimeGetCurrent() - flowLMStart) * 1000
            flowLMTotalMs += flowLMMs

            // EOS detection
            if eosLogit > PocketTtsConstants.eosThreshold && eosStep == nil {
                eosStep = step
                if PocketTtsConstants.detailedTimingLogsEnabled {
                    Self.logger.info("Session chunk \(chunkIndex) EOS at step \(step)")
                }
            }
            if let eos = eosStep, step >= eos + totalFramesAfterEos {
                break
            }

            // Flow decode with actor-isolated RNG
            let flowDecodeStart = CFAbsoluteTimeGetCurrent()
            var localRng = rng
            let latent = try await PocketTtsSynthesizer.flowDecode(
                transformerOut: transformerOut,
                numSteps: PocketTtsConstants.numLsdSteps,
                temperature: temperature,
                model: flowModel,
                rng: &localRng,
                scratch: flowScratch
            )
            rng = localRng
            let flowDecodeMs = (CFAbsoluteTimeGetCurrent() - flowDecodeStart) * 1000
            flowDecodeTotalMs += flowDecodeMs

            // Mimi decode with actor-isolated state
            let mimiStart = CFAbsoluteTimeGetCurrent()
            var localMimi = mimiState
            let frameSamples = try await PocketTtsSynthesizer.runMimiDecoder(
                latent: latent,
                state: &localMimi,
                model: mimiModel,
                mimiKeys: mimiKeys
            )
            mimiState = localMimi
            let mimiMs = (CFAbsoluteTimeGetCurrent() - mimiStart) * 1000
            mimiTotalMs += mimiMs

            let frameTotalMs = (CFAbsoluteTimeGetCurrent() - frameStart) * 1000
            frameCount += 1

            // Detailed frame timings are useful while profiling, but costly in
            // DEBUG app runs because AppLogger mirrors each line to stderr.
            if PocketTtsConstants.detailedTimingLogsEnabled && (step < 3 || step % 10 == 0) {
                Self.logger.info(
                    "[Timing] chunk=\(chunkIndex) frame=\(step) total=\(String(format: "%.1f", frameTotalMs))ms flowLM=\(String(format: "%.1f", flowLMMs))ms flowDec=\(String(format: "%.1f", flowDecodeMs))ms mimi=\(String(format: "%.1f", mimiMs))ms"
                )
            }

            // TTFA: time-to-first-audio from enqueue
            if isFirstFrame {
                isFirstFrame = false
                if PocketTtsConstants.timingLogsEnabled {
                    let ttfaMs = (CFAbsoluteTimeGetCurrent() - enqueueTime) * 1000
                    let sinceSessionMs = (CFAbsoluteTimeGetCurrent() - sessionStartTime) * 1000
                    Self.logger.notice(
                        "[Timing] TTFA=\(String(format: "%.0f", ttfaMs))ms sinceSessionStart=\(String(format: "%.0f", sinceSessionMs))ms"
                    )
                }
            }

            // Yield frame
            frameContinuation.yield(
                PocketTtsSynthesizer.AudioFrame(
                    samples: frameSamples,
                    frameIndex: step,
                    chunkIndex: chunkIndex,
                    chunkCount: chunkCount,
                    utteranceIndex: utteranceIndex
                )
            )

            // Autoregressive feedback
            sequence.writeLatent(latent)
        }

        // Chunk summary
        let chunkMs = (CFAbsoluteTimeGetCurrent() - chunkStart) * 1000
        let avgFrameMs = frameCount > 0 ? chunkMs / Double(frameCount) : 0
        let avgFlowLM = frameCount > 0 ? flowLMTotalMs / Double(frameCount) : 0
        let avgFlowDec = frameCount > 0 ? flowDecodeTotalMs / Double(frameCount) : 0
        let avgMimi = frameCount > 0 ? mimiTotalMs / Double(frameCount) : 0
        let rtf = frameCount > 0 ? (Double(frameCount) * 0.08) / (chunkMs / 1000) : 0
        if PocketTtsConstants.timingLogsEnabled {
            Self.logger.notice(
                "[Timing] chunk=\(chunkIndex) done frames=\(frameCount) total=\(String(format: "%.0f", chunkMs))ms avg/frame=\(String(format: "%.1f", avgFrameMs))ms (flowLM=\(String(format: "%.1f", avgFlowLM)) flowDec=\(String(format: "%.1f", avgFlowDec)) mimi=\(String(format: "%.1f", avgMimi))) RTFx=\(String(format: "%.2f", rtf))"
            )
        }
    }
}
