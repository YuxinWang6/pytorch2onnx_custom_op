#include <torch/script.h>
#include <torch/extension.h>

torch::Tensor neg_tensor(torch::Tensor input) {
    auto neg = input.clamp_max(0);
    return neg;
}

static auto registry = torch::RegisterOperators("usercustom::neg_tensor", &neg_tensor);


