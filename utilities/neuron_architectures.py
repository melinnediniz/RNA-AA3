import math

def test_alphas(n_i, n_o, alphas=[0.5, 2, 3]):
    #alphas = [0.5, 2, 3]
    n_h = []

    for alpha in alphas:
        n_h.append(round(alpha * math.sqrt(n_i * n_o)))
    
    return n_h

def get_architectures(Nh, quant):
    S = [i for i in range(int(Nh))]
    arch_list = []
    for i in range(len(S)):
        for j in range(len(S)):
            if ((S[i], S[j]) not in arch_list) and (i + j == len(S) and (len(arch_list) < quant)):
                arch_list.append((S[i], S[j]))
    
    return arch_list

def create_architectures(n_per_layer: list, n_h):
    layer_arch = []
    layer_0 = get_architectures(n_h[0], n_per_layer[0])
    layer_1 = get_architectures(n_h[1], n_per_layer[1])
    layer_2 = get_architectures(n_h[2], n_per_layer[2])
    layer_arch = layer_0 + layer_1 + layer_2
    
    return layer_arch