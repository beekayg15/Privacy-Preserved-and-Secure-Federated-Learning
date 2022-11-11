import tenseal as ts
import torch
def context():
    context = ts.context(ts.SCHEME_TYPE.CKKS,poly_modulus_degree=8192,coeff_mod_bit_sizes=[60,40,40,60])
    context.global_scale = pow(2, 40)
    context.generate_galois_keys()
    return context

public_key = context()


a = ts.ckks_tensor(public_key, ts.PlainTensor([2,4,6]))
a = a+ts.PlainTensor(0.75)
print(a.decrypt().tolist())