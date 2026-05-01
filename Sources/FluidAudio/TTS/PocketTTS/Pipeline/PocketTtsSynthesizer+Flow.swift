@preconcurrency import CoreML
import Foundation

extension PocketTtsSynthesizer {

    /// Run the flow decoder using Euler integration (LSD steps).
    ///
    /// Converts the 1024-d transformer hidden state into a 32-d audio latent code.
    /// Flow matching works by starting from random Gaussian noise and iteratively
    /// moving it toward a valid audio code over `numSteps` Euler steps. The
    /// transformer output guides each step by predicting a velocity field.
    static func flowDecode(
        transformerOut: MLMultiArray,
        numSteps: Int,
        temperature: Float,
        model: MLModel,
        rng: inout some RandomNumberGenerator
    ) async throws -> [Float] {
        let scratch = try FlowDecoderScratch(model: model)
        return try await flowDecode(
            transformerOut: transformerOut,
            numSteps: numSteps,
            temperature: temperature,
            model: model,
            rng: &rng,
            scratch: scratch
        )
    }

    static func flowDecode(
        transformerOut: MLMultiArray,
        numSteps: Int,
        temperature: Float,
        model: MLModel,
        rng: inout some RandomNumberGenerator,
        scratch: FlowDecoderScratch
    ) async throws -> [Float] {
        let latentDim = PocketTtsConstants.latentDim
        let dt: Float = 1.0 / Float(numSteps)

        // Initialize latent with scaled random noise.
        // sqrt(temperature) because variance scales quadratically with the multiplier.
        var latent = [Float](repeating: 0, count: latentDim)
        let scale = sqrtf(temperature)
        for i in 0..<latentDim {
            latent[i] = Float.gaussianRandom(using: &rng) * scale
        }

        scratch.writeTransformerOut(transformerOut)

        // Euler integration: 8 steps from t=0 to t=1
        for step in 0..<numSteps {
            let sValue = Float(step) * dt
            let tValue = Float(step + 1) * dt

            try await applyFlowDecoderStep(
                latent: &latent,
                s: sValue,
                t: tValue,
                dt: dt,
                model: model,
                scratch: scratch
            )
        }

        return latent
    }

    // MARK: - Private

    /// Run a single flow decoder step.
    ///
    /// - Parameters:
    ///   - s: Start time of this Euler interval (e.g., 0.0, 0.125, 0.25, ...).
    ///   - t: End time of this Euler interval (e.g., 0.125, 0.25, 0.375, ...).
    ///
    /// The model predicts a velocity vector given the current noisy latent and time
    /// interval. The caller applies the Euler update: `latent += velocity * dt`.
    private static func applyFlowDecoderStep(
        latent: inout [Float],
        s: Float,
        t: Float,
        dt: Float,
        model: MLModel,
        scratch: FlowDecoderScratch
    ) async throws {
        let latentDim = PocketTtsConstants.latentDim

        scratch.writeLatent(latent)
        scratch.sArray[0] = NSNumber(value: s)
        scratch.tArray[0] = NSNumber(value: t)

        let output = try await model.compatPrediction(
            from: scratch.inputProvider,
            options: scratch.predictionOptions
        )

        guard let velocityArray = output.featureValue(for: scratch.velocityOutputName)?.multiArrayValue else {
            throw PocketTTSError.processingFailed("Missing flow decoder velocity output")
        }

        let velocityPtr = velocityArray.dataPointer.bindMemory(to: Float.self, capacity: latentDim)
        for i in 0..<latentDim {
            latent[i] += velocityPtr[i] * dt
        }
    }

    final class FlowDecoderScratch: @unchecked Sendable {
        let transformerFlat: MLMultiArray
        let latentArray: MLMultiArray
        let sArray: MLMultiArray
        let tArray: MLMultiArray
        let inputProvider: MLDictionaryFeatureProvider
        let predictionOptions = MLPredictionOptions()
        let velocityOutputName: String

        init(model: MLModel) throws {
            let transformerDim = PocketTtsConstants.transformerDim
            let latentDim = PocketTtsConstants.latentDim

            transformerFlat = try MLMultiArray(
                shape: [1, NSNumber(value: transformerDim)], dataType: .float32)

            latentArray = try MLMultiArray(
                shape: [1, NSNumber(value: latentDim)], dataType: .float32)
            sArray = try MLMultiArray(shape: [1, 1], dataType: .float32)
            tArray = try MLMultiArray(shape: [1, 1], dataType: .float32)

            inputProvider = try MLDictionaryFeatureProvider(dictionary: [
                "transformer_out": transformerFlat,
                "latent": latentArray,
                "s": sArray,
                "t": tArray,
            ])

            guard let velocityOutputName = model.modelDescription.outputDescriptionsByName.keys.first else {
                throw PocketTTSError.processingFailed("Missing flow decoder output name")
            }
            self.velocityOutputName = velocityOutputName
        }

        func writeTransformerOut(_ transformerOut: MLMultiArray) {
            let transformerDim = PocketTtsConstants.transformerDim
            let srcPtr = transformerOut.dataPointer.bindMemory(to: Float.self, capacity: transformerDim)
            let dstPtr = transformerFlat.dataPointer.bindMemory(to: Float.self, capacity: transformerDim)
            dstPtr.update(from: srcPtr, count: transformerDim)
        }

        func writeLatent(_ latent: [Float]) {
            let latentDim = PocketTtsConstants.latentDim
            let latentPtr = latentArray.dataPointer.bindMemory(to: Float.self, capacity: latentDim)
            latent.withUnsafeBufferPointer { buffer in
                guard let base = buffer.baseAddress else { return }
                latentPtr.update(from: base, count: latentDim)
            }
        }
    }

}

// MARK: - Seeded Random

/// Simple seeded random number generator (xoshiro256**).
///
/// Provides reproducible random sequences when a seed is set,
/// and falls back to system entropy when unseeded.
struct SeededRNG: RandomNumberGenerator {
    private var state: (UInt64, UInt64, UInt64, UInt64)

    init(seed: UInt64) {
        // SplitMix64 to expand seed into 4-part state
        var s = seed
        func next() -> UInt64 {
            s &+= 0x9E37_79B9_7F4A_7C15
            var z = s
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        state = (next(), next(), next(), next())
    }

    mutating func next() -> UInt64 {
        let result = rotl(state.1 &* 5, 7) &* 9
        let t = state.1 << 17
        state.2 ^= state.0
        state.3 ^= state.1
        state.1 ^= state.2
        state.0 ^= state.3
        state.2 ^= t
        state.3 = rotl(state.3, 45)
        return result
    }

    private func rotl(_ x: UInt64, _ k: Int) -> UInt64 {
        (x << k) | (x >> (64 - k))
    }
}

extension Float {
    /// Generate a single sample from the standard normal distribution (Box-Muller transform).
    static func gaussianRandom(using rng: inout some RandomNumberGenerator) -> Float {
        let u1 = Float.random(in: Float.leastNonzeroMagnitude...1.0, using: &rng)
        let u2 = Float.random(in: 0.0...1.0, using: &rng)
        return sqrtf(-2.0 * logf(u1)) * cosf(2.0 * .pi * u2)
    }
}
