// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

use ort::environment::Environment;
#[allow(unused)]
use ort::ep::*;
use std::sync::OnceLock;

static ORT_ENV: OnceLock<Result<Environment, String>> = OnceLock::new();

/// Return a handle to the process-wide ONNX Runtime environment.
///
/// The environment is retained for the process lifetime because `ort` permits only one
/// environment per process. Clones share the same underlying environment.
pub fn get_env() -> Result<Environment, String> {
    match ORT_ENV.get_or_init(|| build_env().map_err(|error| error.to_string())) {
        Ok(env) => Ok(env.clone()),
        Err(error) => Err(error.clone()),
    }
}

fn build_env() -> ort::Result<Environment> {
    #[cfg(feature = "backend-candle")]
    ort::set_api(ort_candle::api());
    #[cfg(feature = "backend-tract")]
    ort::set_api(ort_tract::api());

    #[cfg(all(not(feature = "backend-candle"), not(feature = "backend-tract")))]
    ort::init()
        .with_execution_providers([
            #[cfg(feature = "tensorrt")]
            TensorRT::default().build(),
            #[cfg(feature = "cuda")]
            CUDA::default().build(),
            #[cfg(feature = "onednn")]
            OneDNN::default().build(),
            #[cfg(feature = "acl")]
            ACL::default().build(),
            #[cfg(feature = "openvino")]
            OpenVINO::default().build(),
            #[cfg(feature = "coreml")]
            CoreML::default().build(),
            #[cfg(feature = "rocm")]
            ROCm::default().build(),
            #[cfg(feature = "cann")]
            CANN::default().build(),
            #[cfg(feature = "directml")]
            DirectML::default().build(),
            #[cfg(feature = "tvm")]
            TVM::default().build(),
            #[cfg(feature = "nnapi")]
            NNAPI::default().build(),
            #[cfg(feature = "qnn")]
            QNN::default().build(),
            #[cfg(feature = "xnnpack")]
            XNNPACK::default().build(),
            #[cfg(feature = "armnn")]
            ArmNN::default().build(),
            #[cfg(feature = "migraphx")]
            MIGraphX::default().build(),
            #[cfg(feature = "vitis")]
            VitisAI::default().build(),
            #[cfg(feature = "rknpu")]
            RKNPU::default().build(),
            #[cfg(feature = "webgpu")]
            WebGPU::default().build(),
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::get_env;

    #[test]
    fn concurrent_calls_share_one_environment() {
        let workers: Vec<_> = (0..8).map(|_| std::thread::spawn(get_env)).collect();

        for worker in workers {
            worker
                .join()
                .expect("environment initializer thread panicked")
                .expect("environment initialization failed");
        }
    }
}
