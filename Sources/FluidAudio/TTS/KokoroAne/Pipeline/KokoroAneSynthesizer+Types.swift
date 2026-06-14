import Foundation

/// Per-stage wall-clock timings (milliseconds) for one synthesis call.
public struct KokoroAneStageTimings: Sendable, Equatable {
    public var albert: Double = 0
    public var postAlbert: Double = 0
    public var alignment: Double = 0
    public var prosody: Double = 0
    public var noise: Double = 0
    public var vocoder: Double = 0
    public var tail: Double = 0

    /// Sum of all stages, in milliseconds.
    public var totalMs: Double {
        albert + postAlbert + alignment + prosody + noise + vocoder + tail
    }

    public init() {}
}

/// Detailed result of a `KokoroAneManager.synthesizeDetailed` call.
public struct KokoroAneSynthesisResult: Sendable {
    /// 24 kHz mono fp32 PCM samples (raw, not WAV-wrapped).
    public let samples: [Float]
    /// Sample rate (24,000 Hz for the laishere chain).
    public let sampleRate: Int
    /// `T_enc` — phoneme tokens including BOS/EOS.
    public let encoderTokens: Int
    /// `T_a` — acoustic frames produced by PostAlbert / Alignment.
    public let acousticFrames: Int
    /// Per-stage timings.
    public let timings: KokoroAneStageTimings

    /// Convenience: audio duration in seconds.
    public var durationSeconds: Double {
        Double(samples.count) / Double(sampleRate)
    }

    public init(
        samples: [Float],
        sampleRate: Int,
        encoderTokens: Int,
        acousticFrames: Int,
        timings: KokoroAneStageTimings
    ) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.encoderTokens = encoderTokens
        self.acousticFrames = acousticFrames
        self.timings = timings
    }
}

/// Text plus the already-phonemized sequence for one KokoroAne synthesis unit.
public struct KokoroAneTextChunk: Sendable, Equatable {
    /// Original text covered by this synthesis chunk.
    public let text: String
    /// IPA/Bopomofo sequence passed directly into `KokoroAneVocab.encode`.
    public let phonemes: String
    /// Silence to append if another chunk follows this one.
    public let pauseAfterMs: Int

    /// Character count of the phoneme sequence, excluding BOS/EOS.
    public var phonemeCount: Int { phonemes.count }

    public init(text: String, phonemes: String, pauseAfterMs: Int = 0) {
        self.text = text
        self.phonemes = phonemes
        self.pauseAfterMs = pauseAfterMs
    }
}

/// One of the 7 stages in the laishere chain.
public enum KokoroAneStage: String, CaseIterable, Sendable {
    case albert
    case postAlbert
    case alignment
    case prosody
    case noise
    case vocoder
    case tail

    /// `.mlmodelc` filename on disk and on HuggingFace.
    public var bundleName: String {
        switch self {
        case .albert: return "KokoroAlbert.mlmodelc"
        case .postAlbert: return "KokoroPostAlbert.mlmodelc"
        case .alignment: return "KokoroAlignment.mlmodelc"
        case .prosody: return "KokoroProsody.mlmodelc"
        case .noise: return "KokoroNoise_v2.mlmodelc"  // v2: atan2 phase-correction (HF-noise fix)
        case .vocoder: return "KokoroVocoder.mlmodelc"
        case .tail: return "KokoroTail.mlmodelc"
        }
    }
}
