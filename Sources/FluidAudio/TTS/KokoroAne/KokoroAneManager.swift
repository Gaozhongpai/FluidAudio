import Foundation

/// High-level facade for the Kokoro 82M 7-stage CoreML chain
/// (ANE-resident, derived from [laishere/kokoro-coreml](https://github.com/laishere/kokoro-coreml)).
///
/// Splits the model into 7 CoreML graphs with per-stage compute-unit
/// placement (``KokoroAneComputeUnits``). Multi-graph splitting yields a large
/// RTFx win over a single-graph CPU+GPU Kokoro implementation, while MiloFlow's
/// fork keeps phoneme-aware chunking for lower first-audio latency and natural
/// Mandarin punctuation pauses.
///
/// Constraints:
///   * One default voice per variant (`af_heart` for English, `zf_001` for
///     Mandarin); additional voices download on demand via ``setDefaultVoice``
///     / `voice:` / `initialize(preloadVoices:)`.
///   * IPA/Bopomofo input capped at 512 tokens per CoreML pass; longer prompts
///     are split into phoneme-safe chunks.
///   * Loads from HF path `kokoro-82m-coreml/ANE/` (English) or
///     `ANE-zh/` (Mandarin).
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
///   * Text → IPA via ``KokoroAneEnglishPhonemizer`` (Misaki lexicon first
///     — weak function-word forms, vocab punctuation kept as prosody
///     tokens — with per-word BART `G2PModel` fallback for OOV words), or
///     Mandarin text → Bopomofo via ``MandarinG2P``.
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
    private static let sentenceBoundaryCharacters: Set<Character> = [
        ".", "!", "?", "…", "。", "！", "？"
    ]
    private static let clauseBoundaryCharacters: Set<Character> = [
        ",", ";", ":", "，", "；", "：", "、", "—", "–"
    ]
    private static let trailingBoundaryCharacters: Set<Character> = [
        "\"", "'", "”", "’", ")", "]", "}", "）", "】", "」", "』", "》"
    ]
    private let store: KokoroAneModelStore
    private let variant: KokoroAneVariant
    private var defaultVoice: String

    /// English frontend: Misaki lexicon + custom overrides + punctuation
    /// pass-through. Built lazily (needs the chain vocab + lexicon asset);
    /// cached only after a successful lexicon load so a transient download
    /// failure doesn't pin the degraded G2P-only path for the session.
    private var englishPhonemizer: KokoroAneEnglishPhonemizer?
    private var englishCustomLexicon: [String: String] = [:]
    private let englishLexiconCache = LexiconAssetCache()

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
            // Best-effort pre-fetch of the Misaki lexicon cache (weak
            // function-word forms, issue #691). Missing lexicon degrades
            // to the BART-G2P-only path rather than failing initialize.
            _ = await KokoroAneResourceDownloader.ensureEnglishLexicon(directory: nil)
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

    /// Install (or clear) a user-supplied Mandarin pronunciation override.
    ///
    /// Slots in **at the front** of ``MandarinG2P``'s segmentation cascade:
    /// longest-prefix match against the user lexicon runs before the
    /// bundled `pinyin_phrases.bin` / `pinyin_single.bin` lookup. User
    /// entries of equal length to a dict entry win. Pinyin-form tokens
    /// (`zi4`) participate in tone sandhi with surrounding context;
    /// `@`-bopomofo tokens (`@ㄈㄨ4`) bypass sandhi.
    ///
    /// Pass ``MandarinCustomLexicon/empty`` to clear. Only meaningful
    /// for ``KokoroAneVariant/mandarin`` — calling on the English variant
    /// stores the value but has no synthesis effect.
    public func setMandarinCustomLexicon(_ lexicon: MandarinCustomLexicon) async {
        await store.setMandarinCustomLexicon(lexicon)
    }

    /// Install (or clear) a user-supplied English pronunciation override.
    ///
    /// Entries map a word to a Misaki-style IPA string (e.g.
    /// `["to": "tə", "GIF": "ʤˈɪf"]`). The exact spelling is checked
    /// first, then the lower-cased form, before the bundled Misaki
    /// lexicon and the BART G2P fallback. Pass `[:]` to clear.
    ///
    /// Only meaningful for ``KokoroAneVariant/english`` — calling on the
    /// Mandarin variant stores the value but has no synthesis effect
    /// (use ``setMandarinCustomLexicon(_:)`` there).
    public func setEnglishCustomLexicon(_ entries: [String: String]) {
        englishCustomLexicon = entries
        // Rebuild the cached frontend with the new overrides on next use.
        englishPhonemizer = nil
    }

    /// Drop loaded mlmodelcs + voice packs. The store reloads on next call.
    public func cleanup() async {
        await store.cleanup()
        englishPhonemizer = nil
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
        guard !chunks.isEmpty else {
            throw KokoroAneError.inputProcessingFailed("(empty input)")
        }

        var samples: [Float] = []
        var timings = KokoroAneStageTimings()
        var encoderTokens = 0
        var acousticFrames = 0

        for (index, chunk) in chunks.enumerated() {
            let result = try await runChain(phonemes: chunk.phonemes, voice: voice, speed: speed)
            samples.append(contentsOf: result.samples)
            if index < chunks.count - 1 {
                samples.append(contentsOf: Self.pauseSamples(milliseconds: chunk.pauseAfterMs))
            }
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

    /// Resolve the exact phoneme string ``synthesize(text:voice:speed:)``
    /// would feed the 7-stage chain for a single text span — for diagnostics,
    /// tests, and caller-side phoneme caching (issue #691).
    ///
    /// English: Misaki-lexicon-first with BART G2P fallback. Mandarin:
    /// the ``MandarinG2P`` pipeline for Hanzi input, pass-through for
    /// strings that already look like phonemes.
    public func phonemes(for text: String) async throws -> String {
        try await phonemizeForCurrentVariant(text: normalizedChunkSource(text))
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
            chunks.append(
                KokoroAneTextChunk(
                    text: text,
                    phonemes: currentPhonemes,
                    pauseAfterMs: Self.pauseAfterMs(for: text)
                )
            )
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
                pieces.append(
                    KokoroAneTextChunk(
                        text: pieceText,
                        phonemes: phonemes,
                        pauseAfterMs: Self.pauseAfterMs(for: pieceText)
                    )
                )
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
            if candidatePhonemes.count > currentLimit(), !current.isEmpty,
                !(Self.chunkSoftBreakCharacters.contains(character)
                    && candidatePhonemes.count <= KokoroAneConstants.maxPhonemeLength)
            {
                let textToFlush = current.trimmingCharacters(in: .whitespacesAndNewlines)
                current.removeAll(keepingCapacity: true)
                if !textToFlush.isEmpty {
                    let phonemes = try await phonemizeForCurrentVariant(text: textToFlush)
                    if !phonemes.isEmpty {
                        chunks.append(
                            KokoroAneTextChunk(
                                text: textToFlush,
                                phonemes: phonemes,
                                pauseAfterMs: Self.pauseAfterMs(for: textToFlush)
                            )
                        )
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
                chunks.append(
                    KokoroAneTextChunk(
                        text: textToFlush,
                        phonemes: phonemes,
                        pauseAfterMs: Self.pauseAfterMs(for: textToFlush)
                    )
                )
            }
        }
        return chunks.isEmpty ? [chunk] : chunks
    }

    static func pauseAfterMs(for text: String) -> Int {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return 0 }

        for character in trimmed.reversed() {
            if trailingBoundaryCharacters.contains(character) {
                continue
            }
            if sentenceBoundaryCharacters.contains(character) {
                return KokoroAneConstants.pauseSentenceMs
            }
            if clauseBoundaryCharacters.contains(character) {
                return KokoroAneConstants.pauseClauseMs
            }
            return 0
        }
        return 0
    }

    private static func pauseSamples(milliseconds: Int) -> [Float] {
        guard milliseconds > 0 else { return [] }
        let count = (milliseconds * KokoroAneConstants.sampleRate) / 1_000
        return Array(repeating: 0, count: count)
    }

    private func phonemizeForCurrentVariant(text: String) async throws -> String {
        switch variant {
        case .english:
            return try await phonemize(text: text)
        case .mandarin:
            if MandarinG2P.looksLikeHanzi(text) {
                try await store.loadIfNeeded()
                let g2p = try await store.mandarinG2PPipeline()
                return try await g2p.phonemize(text)
            } else {
                // No Hanzi present → caller already supplied bopomofo /
                // ASCII punctuation. Pass through so power users can still
                // override pronunciation manually.
                return text
            }
        }
    }

    /// English text → Misaki-style IPA. Lexicon-first resolution (weak
    /// function-word forms — `to` → `tu`, not the stressed BART citation
    /// form `tˈO`, issue #691), per-word BART G2P fallback for OOV words,
    /// and vocab-supported punctuation kept as prosody/pause tokens.
    private func phonemize(text: String) async throws -> String {
        let phonemizer = await ensureEnglishPhonemizer()
        return try await phonemizer.phonemize(text) { word in
            try await G2PModel.shared.phonemize(word: word)
        }
    }

    /// Build (and cache) the English frontend: chain vocab → allowed
    /// token/punctuation sets, Misaki lexicon cache → weak-form maps.
    /// On any failure returns a transient G2P-only frontend (current
    /// pre-#691 behavior) without caching it, so the lexicon is retried
    /// on the next call.
    private func ensureEnglishPhonemizer() async -> KokoroAneEnglishPhonemizer {
        if let cached = englishPhonemizer { return cached }

        var lower: [String: [String]] = [:]
        var caseSensitive: [String: [String]] = [:]
        var punctuation: Set<Character> = []
        var lexiconLoaded = false

        do {
            try await store.loadIfNeeded()
            let vocab = try await store.vocabulary()
            // Stress/length marks (ˈ ˌ ː) are Unicode modifier letters, so
            // `isLetter` keeps them out of the punctuation set.
            punctuation = Set(
                vocab.map.keys.filter { !$0.isLetter && !$0.isNumber && !$0.isWhitespace })

            if let kokoroDir = await KokoroAneResourceDownloader.ensureEnglishLexicon(directory: nil) {
                let allowedTokens = Set(vocab.map.keys.map(String.init))
                try await englishLexiconCache.ensureLoaded(
                    kokoroDirectory: kokoroDir, allowedTokens: allowedTokens)
                let maps = await englishLexiconCache.lexicons()
                lower = maps.word
                caseSensitive = maps.caseSensitive
                lexiconLoaded = true
            }
        } catch {
            logger.warning(
                "English lexicon unavailable (\(error.localizedDescription)); using BART G2P only")
        }

        let phonemizer = KokoroAneEnglishPhonemizer(
            wordToPhonemes: lower,
            caseSensitiveWordToPhonemes: caseSensitive,
            customLexicon: englishCustomLexicon,
            allowedPunctuation: punctuation
        )
        if lexiconLoaded {
            englishPhonemizer = phonemizer
        }
        return phonemizer
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
