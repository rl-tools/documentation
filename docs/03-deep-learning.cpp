#include <iostream>

#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_cpu.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

namespace rlt = rl_tools;
using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = typename DEVICE::index_t;

int main() {
    DEVICE device;
    RNG rng;
    TI seed = 1;

    {
        constexpr TI BATCH_SIZE = 1;
        constexpr TI INPUT_DIM = 5;
        constexpr TI OUTPUT_DIM = 5;
        constexpr auto ACTIVATION_FUNCTION = rlt::nn::activation_functions::RELU;

        using LAYER_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, OUTPUT_DIM, ACTIVATION_FUNCTION>;
        using CAPABILITY = rlt::nn::capability::Forward<>;
        using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, INPUT_DIM>;
        rlt::nn::layers::dense::Layer<LAYER_CONFIG, CAPABILITY, INPUT_SHAPE> layer;

        rlt::malloc(device, layer);

        rlt::init(device, rng, seed);
        rlt::init_weights(device, layer, rng);

        rlt::print(device, layer.weights.parameters);
        rlt::print(device, layer.biases.parameters);

        rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, INPUT_DIM>> input;
        rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, OUTPUT_DIM>> output;
        rlt::malloc(device, input);
        rlt::malloc(device, output);
        rlt::randn(device, input, rng);
        rlt::print(device, input);

        typename decltype(layer)::Buffer<BATCH_SIZE> buffer;
        rlt::evaluate(device, layer, input, output, buffer, rng);
        rlt::print(device, output);

        using PARAMETER_TYPE = rlt::nn::parameters::Gradient;
        using CAPABILITY_2 = rlt::nn::capability::Gradient<PARAMETER_TYPE, BATCH_SIZE>;
        using LAYER_2_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, OUTPUT_DIM, ACTIVATION_FUNCTION>;
        rlt::nn::layers::dense::Layer<LAYER_2_CONFIG, CAPABILITY_2, INPUT_SHAPE> layer_2;
        rlt::malloc(device, layer_2);
        rlt::copy(device, device, layer, layer_2);
        rlt::zero_gradient(device, layer_2);

        rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, OUTPUT_DIM>> d_output;
        rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, INPUT_DIM>> d_input;
        rlt::malloc(device, d_input);
        rlt::malloc(device, d_output);
        rlt::randn(device, d_output, rng);

        rlt::forward(device, layer_2, input, buffer, rng);
        std::cout << "Output (should be identical to layer_1): " << std::endl;
        rlt::print(device, layer_2.output);
        rlt::backward_full(device, layer_2, input, d_output, d_input, buffer);
        std::cout << "Derivative with respect to the input: " << std::endl;
        rlt::print(device, d_input);

        rlt::print(device, layer_2.weights.gradient);
        rlt::print(device, layer_2.biases.gradient);

        rlt::free(device, layer);
        rlt::free(device, layer_2);
        rlt::free(device, input);
        rlt::free(device, output);
        rlt::free(device, d_input);
        rlt::free(device, d_output);
    }

    {
        constexpr TI BATCH_SIZE = 1;
        constexpr TI INPUT_DIM_MLP = 5;
        constexpr TI OUTPUT_DIM_MLP = 1;
        constexpr TI NUM_LAYERS = 3;
        constexpr TI HIDDEN_DIM = 10;
        constexpr auto ACTIVATION_FUNCTION_MLP = rlt::nn::activation_functions::RELU;
        constexpr auto OUTPUT_ACTIVATION_FUNCTION_MLP = rlt::nn::activation_functions::IDENTITY;

        using MODEL_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, OUTPUT_DIM_MLP, NUM_LAYERS, HIDDEN_DIM, ACTIVATION_FUNCTION_MLP, OUTPUT_ACTIVATION_FUNCTION_MLP>;
        using PARAMETER_TYPE = rlt::nn::parameters::Adam;
        using CAPABILITY = rlt::nn::capability::Gradient<PARAMETER_TYPE>;
        using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI>;
        using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
        using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, INPUT_DIM_MLP>;
        using MODEL_TYPE = rlt::nn_models::mlp::NeuralNetwork<MODEL_CONFIG, CAPABILITY, INPUT_SHAPE>;

        OPTIMIZER optimizer;
        MODEL_TYPE model;
        typename MODEL_TYPE::Buffer<> buffer;

        rlt::malloc(device, model);
        rlt::malloc(device, optimizer);
        rlt::init_weights(device, model, rng);
        rlt::zero_gradient(device, model);
        rlt::reset_optimizer_state(device, optimizer, model);
        rlt::malloc(device, buffer);

        rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, INPUT_DIM_MLP>> input_mlp, d_input_mlp;
        rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, OUTPUT_DIM_MLP>> d_output_mlp;
        rlt::malloc(device, input_mlp);
        rlt::malloc(device, d_input_mlp);
        rlt::malloc(device, d_output_mlp);

        rlt::randn(device, input_mlp, rng);
        rlt::forward(device, model, input_mlp, buffer, rng);
        T output_value = rlt::get(model.output_layer.output, 0, 0);
        std::cout << output_value << std::endl;

        T target_output_value = 1;
        T error = target_output_value - output_value;
        rlt::set(d_output_mlp, 0, 0, -error);
        rlt::backward(device, model, input_mlp, d_output_mlp, buffer);

        rlt::step(device, optimizer, model);

        rlt::forward(device, model, input_mlp, buffer, rng);
        std::cout << rlt::get(model.output_layer.output, 0, 0) << std::endl;

        for(TI i=0; i < 10000; i++){
            rlt::zero_gradient(device, model);
            T mse = 0;
            for(TI batch_i=0; batch_i < 32; batch_i++){
                rlt::randn(device, input_mlp, rng);
                rlt::forward(device, model, input_mlp, buffer, rng);
                T output_value = rlt::get(model.output_layer.output, 0, 0);
                T target_output_value = rlt::max(device, input_mlp);
                T error = target_output_value - output_value;
                rlt::set(d_output_mlp, 0, 0, -error);
                rlt::backward(device, model, input_mlp, d_output_mlp, buffer);
                mse += error * error;
            }
            rlt::step(device, optimizer, model);
            if(i % 1000 == 0)
                std::cout << "Squared error: " << mse/32 << std::endl;
        }

        rlt::set(input_mlp, 0, 0, +0.0);
        rlt::set(input_mlp, 0, 1, -0.1);
        rlt::set(input_mlp, 0, 2, +0.5);
        rlt::set(input_mlp, 0, 3, -0.4);
        rlt::set(input_mlp, 0, 4, +0.1);

        rlt::forward(device, model, input_mlp, buffer, rng);
        std::cout << rlt::get(model.output_layer.output, 0, 0) << std::endl;

        for(TI i=0; i < 10; i++){
            rlt::randn(device, input_mlp, rng);
            rlt::forward(device, model, input_mlp, buffer, rng);
            std::cout << "max: " << rlt::max(device, input_mlp) << " output: " << rlt::get(model.output_layer.output, 0, 0) << std::endl;
        }
    }

    {
        using namespace rlt::nn_models::sequential;

        constexpr TI BATCH_SIZE = 1;
        constexpr TI INPUT_DIM_MLP = 5;

        using LAYER_1_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 32, rlt::nn::activation_functions::ActivationFunction::RELU>;
        using LAYER_1 = rlt::nn::layers::dense::BindConfiguration<LAYER_1_CONFIG>;
        using LAYER_2_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 16, rlt::nn::activation_functions::ActivationFunction::RELU>;
        using LAYER_2 = rlt::nn::layers::dense::BindConfiguration<LAYER_2_CONFIG>;
        using LAYER_3_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
        using LAYER_3 = rlt::nn::layers::dense::BindConfiguration<LAYER_3_CONFIG>;

        using MODULE_CHAIN = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;

        using CAPABILITY = rlt::nn::capability::Forward<>;
        constexpr TI SEQUENCE_LENGTH = 1;
        using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, INPUT_DIM_MLP>;

        using SEQUENTIAL = Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;

        SEQUENTIAL sequential;
        SEQUENTIAL::Buffer<> sequential_buffer;
        rlt::malloc(device, sequential);
        rlt::malloc(device, sequential_buffer);

        rlt::init_weights(device, sequential, rng);

        rlt::Tensor<rlt::tensor::Specification<T, TI, typename SEQUENTIAL::INPUT_SHAPE, false>> input;
        rlt::Tensor<rlt::tensor::Specification<T, TI, typename SEQUENTIAL::OUTPUT_SHAPE, false>> output;
        rlt::randn(device, input, rng);

        rlt::evaluate(device, sequential, input, output, sequential_buffer, rng);
        rlt::print(device, output);

        using PARAMETER_TYPE = rlt::nn::parameters::Adam;
        using NEW_CAPABILITY = rlt::nn::capability::Gradient<PARAMETER_TYPE, false>;
        using NEW_SEQUENTIAL_CAPABILITY = SEQUENTIAL::CHANGE_CAPABILITY<NEW_CAPABILITY>;

        constexpr TI NEW_BATCH_SIZE = 32;

        using NEW_SEQUENTIAL = NEW_SEQUENTIAL_CAPABILITY::CHANGE_BATCH_SIZE<TI, NEW_BATCH_SIZE>;

        NEW_SEQUENTIAL new_sequential;
        NEW_SEQUENTIAL::Buffer<false> new_sequential_buffer;
        rlt::copy(device, device, sequential, new_sequential);

        rlt::evaluate(device, new_sequential, input, output, new_sequential_buffer, rng);
        rlt::print(device, output);

        rlt::Tensor<rlt::tensor::Specification<T, TI, typename NEW_SEQUENTIAL::INPUT_SHAPE, false>> new_input;
        rlt::randn(device, new_input, rng);

        rlt::forward(device, new_sequential, new_input, new_sequential_buffer, rng);

        rlt::Tensor<rlt::tensor::Specification<T, TI, typename NEW_SEQUENTIAL::OUTPUT_SHAPE, false>> new_d_output;
        rlt::randn(device, new_d_output, rng);

        rlt::zero_gradient(device, new_sequential);
        rlt::backward(device, new_sequential, new_input, new_d_output, new_sequential_buffer);

        rlt::print(device, new_sequential.content.weights.gradient);
    }

    return 0;
}
