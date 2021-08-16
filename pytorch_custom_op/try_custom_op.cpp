#include <torch/script.h>
#include <torch/extension.h>

torch::Tensor try_ncrelu_forward(torch::Tensor input) {
    auto neg = input.clamp_max(0);
    return neg;
}

static auto registry = torch::RegisterOperators("usercustom::try_ncrelu_forward", &try_ncrelu_forward);


