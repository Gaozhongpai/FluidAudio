import Foundation

/// High-level facade for the Kokoro 82M 7-stage CoreML chain
/// (ANE-resident, derived from [laishere/kokoro-coreml](https://github.com/laishere/kokoro-coreml)).
///
/// Splits the model so ANE-friendly layers (Albert / PostAlbert / Alignment /
/// Vocoder) stay resident on the Neural Engine while Prosody / Noise / Tail
/// run on CPU+GPU. Yields **3-11× RTFx** on Apple Silicon vs. the single-graph
/// ``KokoroTtsManager``.
///
/// Trade-offs vs. ``KokoroTtsManager``:
///
/// |                  | ``KokoroTtsManager``      | ``KokoroAneManager``         |
/// |------------------|---------------------------|------------------------------|
/// | Compute          | CPU + GPU                 | 4 stages on ANE, 3 on GPU    |
/// | Voices           | Multi (`.json` packs)     | Single (`af_heart.bin`)      |
/// | Long input       | Built-in chunker          | Phoneme-aware chunking       |
/// | Custom lexicon   | Yes (`TtsCustomLexicon`)  | No                           |
/// | HF path          | `kokoro-82m-coreml/`      | `kokoro-82m-coreml/ANE/`     |
///
/// Mirrors the public surface of ``KokoroTtsManager`` so callers can swap
/// backends with minimal churn. Internally:
///   * Text → IPA via the existing `G2PModel` (per-word, joined with " ")
///   * IPA → input ids via `KokoroAneVocab`
///   * Voice pack slice via `KokoroAneVoicePack`
///   * 7 stages via `KokoroAneSynthesizer`
///   * Float samples → WAV via `AudioWAV`
///
/// Concurrency: actor-isolated. `KokoroAneModelStore` is an actor too, so all
/// model access flows through an awaited boundary — no shared mutable state
/// is exposed.
public actor KokoroAneManager {

    private let logger = AppLogger(category: "KokoroAneManager")
    private static let chunkSoftBreakCharacters: Set<Character> = [
        ".", "!", "?", ";", ",", "。", "！", "？", "；", "，", "、", "\n"
    ]
    private let store: KokoroAneModelStore
    private let variant: KokoroAneVariant
    private var defaultVoice: String

    public init(
        variant: KokoroAneVariant = .english,
        defaultVoice: String? = nil,
        directory: URL? = nil,
        computeUnits: KokoroAneComputeUnits = .default,
        modelStore: KokoroAneModelStore? = nil
    ) {
        self.variant = variant
        self.defaultVoice = defaultVoice ?? variant.defaultVoice
        self.store =
            modelStore
            ?? KokoroAneModelStore(
                directory: directory, computeUnits: computeUnits, variant: variant)
    }

    // MARK: - Lifecycle

    /// Download (if missing), load all 7 mlmodelcs + vocab + default voice
    /// pack. Optionally pre-warm additional voice packs.
    public func initialize(preloadVoices: Set<String>? = nil) async throws {
        try await store.loadIfNeeded()
        // English G2P CoreML assets live in the kokoro repo and are loaded
        // from ~/.cache/fluidaudio/Models/kokoro/. The Mandarin variant
        // routes through the in-process MandarinG2P pipeline (loaded by
        // store.loadIfNeeded()) and never calls G2PModel.shared, so the
        // English G2P bundle would just be wasted bandwidth + memory.
        //
        // For English: G2PModel.loadIfNeeded only reads from cache (it
        // never downloads), so first-time KokoroAne users who have never
        // run the regular kokoro backend would otherwise hit a cryptic
        // G2PModelError.vocabLoadFailed. Fetch G2P assets explicitly
        // before warming the in-process G2P model.
        //
        // NOTE: pass nil (not `directory`) — `G2PModel.shared` is a singleton
        // that hardcodes the default cache path (TtsModels.cacheDirectoryURL()
        // /Models/kokoro). If we honoured the caller's custom `directory` here
        // we'd download to a path G2PModel can't see and still hit
        // vocabLoadFailed. The KokoroAne mlmodelc chain itself does respect
        // `directory` (via store), only the shared G2P assets are pinned.
        if variant == .english {
            try await KokoroAneResourceDownloader.ensureG2PAssets(directory: nil)
            try await G2PModel.shared.ensureModelsAvailable()
        }
        if let voices = preloadVoices {
            for voice in voices {
                _ = try await store.voicePack(voice)
            }
        }
    }

    /// `true` once the 7 mlmodelcs + vocab are resident.
    public func isAvailable() async -> Bool {
        await store.isLoaded
    }

    /// Override the voice used by default.
    public func setDefaultVoice(_ voice: String) {
        self.defaultVoice = voice
    }

    /// Drop loaded mlmodelcs + voice packs. The store reloads on next call.
    public func cleanup() async {
        await store.cleanup()
    }

    // MARK: - Synthesis

    /// One-shot text → 24 kHz mono 16-bit PCM WAV.
    public func synthesize(
        text: String,
        voice: String? = nil,
        speed: Float = KokoroAneConstants.defaultSpeed
    ) async throws -> Data {
        let result = try await synthesizeDetailed(text: text, voice: voice, speed: speed)
        return try wavData(from: result)
    }

    /// Text → samples + per-stage timings.
    ///
    /// For ``KokoroAneVariant/mandarin`` the input is routed through
    /// ``MandarinG2P``: Hanzi → forward-max-match segmentation
    /// (`pinyin_phrases.bin` + `pinyin_single.bin`) → diacritic
    /// → tone-digit normalization → 3+3 / 不 / 一 sandhi → bopomofo +
    /// tone-digit string. Strings that already look like phonemes
    /// (no Hanzi) bypass the pipeline and are forwarded as-is, so
    /// callers can still feed pre-computed bopomofo when they want
    /// to override the bundled lexicon.
    public func synthesizeDetailed(
        text: String,
        voice: String? = nil,
        speed: Float = KokoroAneConstants.defaultSpeed
    ) async throws -> KokoroAneSynthesisResult {
        let chunks = try await synthesisChunks(text: text)
        guard let first = chunks.first else {
            throw KokoroAneError.inputProcessingFailed("(empty input)")
        }

        if chunks.count == 1 {
            return try await runChain(phonemes: first.phonemes, voice: voice, speed: speed)
        }

        var samples: [Float] = []
        var timings = KokoroAneStageTimings()
        var encoderTokens = 0
        var acousticFrames = 0

        for chunk in chunks {
            let result = try await runChain(phonemes: chunk.phonemes, voice: voice, speed: speed)
            samples.append(contentsOf: result.samples)
            Self.add(result.timings, to: &timings)
            encoderTokens += result.encoderTokens
            acousticFrames += result.acousticFrames
        }

        return KokoroAneSynthesisResult(
            samples: samples,
            sampleRate: KokoroAneConstants.sampleRate,
            encoderTokens: encoderTokens,
            acousticFrames: acousticFrames,
            timings: timings
        )
    }

    /// Convert text into phoneme-safe synthesis chunks.
    ///
    /// This mirrors Kokoro's generator-style behavior: text is split at natural
    /// punctuation when possible, then bounded by actual IPA/Bopomofo length so
    /// each chunk fits the 510-token CoreML context. The first chunk defaults to
    /// a smaller budget to reduce first-audio latency.
    public func synthesisChunks(
        text: String,
        firstChunkPhonemeLimit: Int = KokoroAneConstants.defaultFirstChunkPhonemeLimit,
        chunkPhonemeLimit: Int = KokoroAneConstants.defaultChunkPhonemeLimit
    ) async throws -> [KokoroAneTextChunk] {
        let firstLimit = Self.clampedChunkLimit(firstChunkPhonemeLimit)
        let regularLimit = Self.clampedChunkLimit(chunkPhonemeLimit)
        let normalized = normalizedChunkSource(text)
        guard !normalized.isEmpty else {
            throw KokoroAneError.inputProcessingFailed("(empty input)")
        }

        let pieces = try await phonemizedPieces(from: normalized)
        guard !pieces.isEmpty else {
            throw KokoroAneError.inputProcessingFailed(
                "G2P produced no phonemes for input '\(text)'")
        }

        var chunks: [KokoroAneTextChunk] = []
        var currentText = ""
        var currentPhonemes = ""

        func flush() {
            let text = currentText.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty, !currentPhonemes.isEmpty else {
                currentText.removeAll(keepingCapacity: true)
                currentPhonemes.removeAll(keepingCapacity: true)
                return
            }
            chunks.append(KokoroAneTextChunk(text: text, phonemes: currentPhonemes))
            currentText.removeAll(keepingCapacity: true)
            currentPhonemes.removeAll(keepingCapacity: true)
        }

        func append(_ piece: KokoroAneTextChunk) {
            currentText.append(piece.text)
            currentPhonemes.append(piece.phonemes)
        }

        for piece in pieces {
            var pending = [piece]
            while !pending.isEmpty {
                let next = pending.removeFirst()
                let targetLimit = chunks.isEmpty ? firstLimit : regularLimit

                if next.phonemeCount > targetLimit {
                    if !currentPhonemes.isEmpty {
                        flush()
                        pending.insert(next, at: 0)
                        continue
                    }

                    let split = try await splitOversizedChunk(
                        next,
                        firstChunkPhonemeLimit: chunks.isEmpty ? firstLimit : regularLimit,
                        chunkPhonemeLimit: regularLimit
                    )
                    if split.count == 1 && split[0] == next {
                        append(next)
                    } else {
                        pending.insert(contentsOf: split, at: 0)
                    }
                    continue
                }

                if currentPhonemes.isEmpty
                    || currentPhonemes.count + next.phonemeCount <= targetLimit
                {
                    append(next)
                } else {
                    flush()
                    pending.insert(next, at: 0)
                }
            }
        }

        flush()
        return chunks
    }

    /// Bypass G2P; feed an already-IPA phoneme string directly.
    ///
    /// For the ``KokoroAneVariant/mandarin`` variant the `phonemes` argument
    /// must be Bopomofo + tone digits + IPA punctuation matching the
    /// `kokoro-82m-coreml/ANE-zh/vocab.json` token set.
    public func synthesizeFromPhonemes(
        _ phonemes: String,
        voice: String? = nil,
        speed: Float = KokoroAneConstants.defaultSpeed
    ) async throws -> Data {
        let result = try await runChain(phonemes: phonemes, voice: voice, speed: speed)
        return try wavData(from: result)
    }

    /// Bypass G2P; return samples + timings.
    public func synthesizeFromPhonemesDetailed(
        _ phonemes: String,
        voice: String? = nil,
        speed: Float = KokoroAneConstants.defaultSpeed
    ) async throws -> KokoroAneSynthesisResult {
        try await runChain(phonemes: phonemes, voice: voice, speed: speed)
    }

    // MARK: - Private

    private func runChain(
        phonemes: String,
        voice: String?,
        speed: Float
    ) async throws -> KokoroAneSynthesisResult {
        try await store.loadIfNeeded()
        let vocab = try await store.vocabulary()
        let voiceName = voice ?? defaultVoice
        let pack = try await store.voicePack(voiceName)

        let inputIds = try vocab.encode(phonemes)
        // Voice pack indexing matches `convert.py:get_ref_data` — row is the
        // raw phoneme-string length (BOS/EOS not counted).
        let phonemeCount = phonemes.count
        let (styleS, styleTimbre) = pack.slice(for: phonemeCount)

        return try await KokoroAneSynthesizer.synthesize(
            inputIds: inputIds,
            styleS: styleS,
            styleTimbre: styleTimbre,
            speed: speed,
            store: store
        )
    }

    private static func add(
        _ rhs: KokoroAneStageTimings,
        to lhs: inout KokoroAneStageTimings
    ) {
        lhs.albert += rhs.albert
        lhs.postAlbert += rhs.postAlbert
        lhs.alignment += rhs.alignment
        lhs.prosody += rhs.prosody
        lhs.noise += rhs.noise
        lhs.vocoder += rhs.vocoder
        lhs.tail += rhs.tail
    }

    private static func clampedChunkLimit(_ limit: Int) -> Int {
        max(16, min(limit, KokoroAneConstants.maxPhonemeLength))
    }

    private func normalizedChunkSource(_ text: String) -> String {
        switch variant {
        case .english:
            text.trimmingCharacters(in: .whitespacesAndNewlines)
        case .mandarin:
            MandarinG2P.normalizeText(text)
        }
    }

    private func phonemizedPieces(from text: String) async throws -> [KokoroAneTextChunk] {
        var pieces: [KokoroAneTextChunk] = []
        for pieceText in Self.softTextPieces(from: text) {
            let phonemes = try await phonemizeForCurrentVariant(text: pieceText)
            if !phonemes.isEmpty {
                pieces.append(KokoroAneTextChunk(text: pieceText, phonemes: phonemes))
            }
        }
        return pieces
    }

    private static func softTextPieces(from text: String) -> [String] {
        var pieces: [String] = []
        var current = ""

        func flush() {
            let piece = current.trimmingCharacters(in: .whitespacesAndNewlines)
            if !piece.isEmpty {
                pieces.append(piece)
            }
            current.removeAll(keepingCapacity: true)
        }

        for character in text {
            current.append(character)
            if chunkSoftBreakCharacters.contains(character) {
                flush()
            }
        }
        flush()
        return pieces
    }

    private func splitOversizedChunk(
        _ chunk: KokoroAneTextChunk,
        firstChunkPhonemeLimit: Int,
        chunkPhonemeLimit: Int
    ) async throws -> [KokoroAneTextChunk] {
        let characters = Array(chunk.text)
        guard characters.count > 1 else { return [chunk] }

        var chunks: [KokoroAneTextChunk] = []
        var current = ""

        func currentLimit() -> Int {
            chunks.isEmpty ? firstChunkPhonemeLimit : chunkPhonemeLimit
        }

        for character in characters {
            let candidate = current + String(character)
            let candidatePhonemes = try await phonemizeForCurrentVariant(text: candidate)
            if candidatePhonemes.count > currentLimit(), !current.isEmpty {
                let textToFlush = current.trimmingCharacters(in: .whitespacesAndNewlines)
                current.removeAll(keepingCapacity: true)
                if !textToFlush.isEmpty {
                    let phonemes = try await phonemizeForCurrentVariant(text: textToFlush)
                    if !phonemes.isEmpty {
                        chunks.append(KokoroAneTextChunk(text: textToFlush, phonemes: phonemes))
                    }
                }
                current = String(character)
            } else {
                current = candidate
            }
        }

        let textToFlush = current.trimmingCharacters(in: .whitespacesAndNewlines)
        current.removeAll(keepingCapacity: true)
        if !textToFlush.isEmpty {
            let phonemes = try await phonemizeForCurrentVariant(text: textToFlush)
            if !phonemes.isEmpty {
                chunks.append(KokoroAneTextChunk(text: textToFlush, phonemes: phonemes))
            }
        }
        return chunks.isEmpty ? [chunk] : chunks
    }

    private func phonemizeForCurrentVariant(text: String) async throws -> String {
        switch variant {
        case .english:
            return try await phonemize(text: text)
        case .mandarin:
            try await store.loadIfNeeded()
            if MandarinG2P.looksLikeHanzi(text) {
                let g2p = try await store.mandarinG2PPipeline()
                return try g2p.phonemize(text)
            } else {
                // No Hanzi present → caller already supplied bopomofo /
                // ASCII punctuation. Pass through so power users can still
                // override pronunciation manually.
                return text
            }
        }
    }

    /// Whitespace-split, per-word G2P, joined with " ". Punctuation is
    /// stripped because the laishere vocab is IPA-only — punctuation chars
    /// would just be dropped at `KokoroAneVocab.encode` anyway.
    private func phonemize(text: String) async throws -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw KokoroAneError.inputProcessingFailed("(empty input)")
        }

        let words = trimmed.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        var parts: [String] = []
        parts.reserveCapacity(words.count)

        for word in words {
            let cleaned = word.trimmingCharacters(in: .punctuationCharacters).lowercased()
            guard !cleaned.isEmpty else { continue }
            do {
                if let ipa = try await G2PModel.shared.phonemize(word: cleaned) {
                    parts.append(ipa.joined())
                } else {
                    logger.warning("G2P returned nil for word '\(cleaned)' — skipping")
                }
            } catch {
                logger.warning(
                    "G2P failed on word '\(cleaned)': \(error.localizedDescription)")
                throw error
            }
        }

        let joined = parts.joined(separator: " ")
        if joined.isEmpty {
            throw KokoroAneError.inputProcessingFailed(
                "G2P produced no phonemes for input '\(trimmed)'")
        }
        return joined
    }

    private func wavData(from result: KokoroAneSynthesisResult) throws -> Data {
        do {
            return try AudioWAV.data(
                from: result.samples,
                sampleRate: Double(result.sampleRate))
        } catch {
            throw KokoroAneError.audioConversionFailed(error.localizedDescription)
        }
    }
}
