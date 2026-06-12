import XCTest

@testable import FluidAudio

final class KokoroAneChunkingTests: XCTestCase {

    func testPauseAfterMsFollowsTrailingPunctuation() {
        XCTAssertEqual(
            KokoroAneManager.pauseAfterMs(for: "你好，"),
            KokoroAneConstants.pauseClauseMs)
        XCTAssertEqual(
            KokoroAneManager.pauseAfterMs(for: "你好。"),
            KokoroAneConstants.pauseSentenceMs)
        XCTAssertEqual(
            KokoroAneManager.pauseAfterMs(for: "你好。”"),
            KokoroAneConstants.pauseSentenceMs)
        XCTAssertEqual(KokoroAneManager.pauseAfterMs(for: "你好"), 0)
    }

    func testMandarinSoftLimitKeepsPunctuationOnPrecedingChunk() async throws {
        let manager = KokoroAneManager(variant: .mandarin)
        let first = String(repeating: "ㄋ", count: 16) + ","
        let second = String(repeating: "ㄏ", count: 16) + "."

        let chunks = try await manager.synthesisChunks(
            text: first + second,
            firstChunkPhonemeLimit: 16,
            chunkPhonemeLimit: 16
        )

        XCTAssertEqual(chunks.map(\.text), [first, second])
        XCTAssertEqual(
            chunks.map(\.pauseAfterMs),
            [KokoroAneConstants.pauseClauseMs, KokoroAneConstants.pauseSentenceMs]
        )
    }
}
