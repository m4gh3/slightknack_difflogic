import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import optax
import functools
import itertools

GATES = 16

def gate_all(a, b):
    return jnp.array([
        # jnp.maximum(0., a)
        jnp.zeros_like(a),
        a * b,
        a - a*b,
        a,
        b - a*b,
        b,
        a + b - 2.0*a*b,
        a + b - a*b,
        1.0 - (a + b - a*b),
        1.0 - (a + b - 2.0*a*b),
        1.0 - b,
       	1.0 - b + a*b,
        1.0 - a,
        1.0 - a + a*b,
        1.0 - a*b,
        jnp.ones_like(a),
    ])

# gate_all(left, right) and w have shape (16, n)
# where n is the dimension of left/right
# This is a batched dot product along the second axis (axis 1)
def gate(left, right, w, hard):
    w_gate = \
        jnp.exp(w) / jnp.sum(jnp.exp(w), axis=0, keepdims=True) if not hard \
        else jax.nn.one_hot(jnp.argmax(w, axis=0), GATES).T
    return jnp.sum(gate_all(left, right) * w_gate, axis=0)

# def relu(left, right, w):
#     return jnp.maximum(0., left)

def gate_normalize(w):
    sum_col = jnp.sum(w, axis=0)
    return w / sum_col[None,:]

# uniform random vectors length 16 whose entries sum to 1
def rand_gate(key, n):
    # return gate_normalize(random.uniform(key, (GATES, n)))
    result = jnp.zeros((GATES, n))
    result = result.at[3, :].set(1.0)
    return result
    # return jnp.full((GATES, n), 1. / GATES)

def rand_layer(key, m, n):
    left_key, right_key, gate_key = random.split(key, 3)
    left  = 2*jax.random.normal(left_key, (n, m) ) 
    right = 2*jax.random.normal(right_key, (n, m) )
    g     = rand_gate(gate_key, n)
    return g, left, right

def rand_network(key, sizes):
    keys = random.split(key, len(sizes))
    dims = zip(keys, sizes[:-1], sizes[1:])
    return list(zip(*[rand_layer(*dim) for dim in dims]))

def hardmax(w):
    m = w.shape[1]
    idx = jnp.argmax(w, axis=1 )
    return jax.nn.one_hot(idx, m )

def predict(params, inp, hard):
    active = inp
    for g, left, right in zip(*params):
        if hard:
            outs_l = jnp.dot(hardmax(left),  active )
            outs_r = jnp.dot(hardmax(right), active )
        else:
            left = 0.5*jax.lax.stop_gradient(hardmax(left)-jax.nn.softmax(left)) + jax.nn.softmax(left)
            right = 0.5*jax.lax.stop_gradient(hardmax(right)-jax.nn.softmax(right)) + jax.nn.softmax(right)
            outs_l = jnp.dot(left,  active)
            outs_r = jnp.dot(right, active)

        active = gate(outs_l, outs_r, g, hard)
    return active

predict_batch = vmap(predict, in_axes=(None, 0, None))


def loss(params, inp, out, hard):
    preds = predict_batch(params, inp, hard)
    return jnp.mean(jnp.square(preds - out))

@functools.partial(jit, static_argnums=(3,))
def update_adamw(params, x, y, opt, opt_state):
    # I think params has to be the first argument?
    grads = grad(loss)(params, x, y, False)
    grads, opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, grads)
    return new_params, opt_state

def conway_kernel(inp):
    def c_and(a, b): return a * b
    def c_or(a, b): return a + b - a * b
    def c_eq(a, b): return jnp.maximum(0.0, 1.0 - jnp.abs(a - b)) # _/\_
    alive = inp[4]
    inp = inp.at[4].set(0)
    neighbors = jnp.sum(inp)
    return c_or(c_eq(3, neighbors), c_and(alive, c_eq(2, neighbors)))

conway_kernel_batch = lambda x: jnp.expand_dims(vmap(conway_kernel)(x), axis=-1)

def conway_draw(inp):
    out = conway_kernel(inp)
    inp = inp.reshape((3, 3))
    for row in inp:
        for x in row:
            print("x" if x > 0.5 else "-", end="")
        print(" X" if out > 0.5 else " _")

def conway_sample(key):
    return jnp.round(random.uniform(key, (9,)))

def conway_sample_batch(key, size):
    keys = random.split(key, size)
    return vmap(conway_sample)(keys)

def conway_sample_all():
    return jnp.array([[float(b) for b in bin(i)[2:].zfill(9)] for i in range(512)])

def train_adamw(key, params, epochs=3001, batch_size=512): #epochs=3001
    import time
    keys = random.split(key, epochs)
    opt = optax.chain(
        optax.clip(100.0), 
        optax.adamw(learning_rate=0.01, b1=0.9, b2=0.99, weight_decay=1e-2), #lr=0.05
        optax.add_noise(0.01, 0.99, 0 ),
    )
    opt_state = opt.init(params)

    x = conway_sample_all()
    y = conway_kernel_batch(x)
    for (i, key_epoch) in enumerate(keys):
        key_train, key_accuracy = random.split(key_epoch)

        time_start = time.time()
        params, opt_state = update_adamw(params, x, y, opt, opt_state)
        time_epoch = time.time() - time_start


        #"partially settle the wiring", reinitialize wiring distribution logits based on which are the strongest connections
        #unfortunately the only way I found to do this requires reinitializing the optimizer as well
        if (i % 4000 == 0) and i > 0:
            new_params = (params[0],[],[])

            for i in range(len(params[1])):
                new_params[1].append(5*hardmax(params[1][i]))
                new_params[2].append(5*hardmax(params[2][i]))

            params = (new_params[0],tuple(new_params[1]),tuple(new_params[2]))

            opt_state = opt.init(params)

        print(f"Epoch ({i+1}/{epochs}) in {time_epoch:.3g} s/epoch", end="   \r")
        if i % 1000 == 0: 
            debug_loss(key_accuracy, params, x, y)



        # if i % 10000 == 0: debug_params(params)
    return params

def debug_loss(key, params, x, y):
    print()
    x_test = conway_sample_all()
    y_test = conway_kernel_batch(x_test)
    train_loss = loss(params, x, y, False)
    test_loss = loss(params, x_test, y_test, False)
    test_loss_hard = loss(params, x_test, y_test, True)
    preds = predict_batch(params, x_test, False)
    preds_hard = predict_batch(params, x_test, True)
    print("[", *[f"{x:.3g}" for x in preds[0:5].flatten().tolist()], "]", preds_hard[0:5].flatten(), y_test[0:5].flatten())
    print(f"train_loss: {train_loss:.3g}", end="; ")
    print(f"test_loss: {test_loss:.3g}", end="; ")
    print(f"test_loss_hard: {test_loss_hard:.3g}")

def debug_params(params):
    for i, param in enumerate(params):
        print("LAYER", i, param.shape)
        for gate in param.T.tolist():
            for logic in gate:
                if logic > 1: print("█ ", end="")
                elif logic < 0: print("▁ ", end="")
                else: print("▄ ", end="")
            print()

def ext_gate_name(idx, l, r):
    names = [
        lambda a, b: "0",
        lambda a, b: f"{a} & {b}",
        lambda a, b: f"{a} & ~{b}",
        lambda a, b: a,
        lambda a, b: f"{b} & ~{a}",
        lambda a, b: b,
        lambda a, b: f"{a} ^ {b}",
        lambda a, b: f"{a} | {b}",
        lambda a, b: f"~({a} | {b})",
        lambda a, b: f"~({a} ^ {b})",
        lambda a, b: f"~{b}",
        lambda a, b: f"{a} | ~{b}",
        lambda a, b: f"~{a}",
        lambda a, b: f"{b} | ~{a}",
        lambda a, b: f"~({a} & {b})",
        lambda a, b: "~0",
    ]
    return names[idx](l, r)

def ext_add_deps(req, idx, l, r):
    deps = [
        lambda a, b: [],
        lambda a, b: [a, b],
        lambda a, b: [a, b],
        lambda a, b: [a],
        lambda a, b: [a, b],
        lambda a, b: [b],
        lambda a, b: [a, b],
        lambda a, b: [a, b],
        lambda a, b: [a, b],
        lambda a, b: [a, b],
        lambda a, b: [b],
        lambda a, b: [a, b],
        lambda a, b: [a],
        lambda a, b: [a, b],
        lambda a, b: [a, b],
        lambda a, b: [],
    ]
    for g in deps[idx](l, r):
        req.add(g)
    return req

def ext_layer(param, left, right, layer):
    out = []
    for i, (g, l, r) in enumerate(zip(param.T, left, right)):
        idx_g = jnp.argmax(g, axis=0)
        idx_l = jnp.argmax(l, axis=0)
        idx_r = jnp.argmax(r, axis=0)
        instr = (f"g_{layer+1}_{i}", idx_g, f"g_{layer}_{idx_l}", f"g_{layer}_{idx_r}")
        if instr is not None:
            out.append(instr)
    return out

def ext_logic(params, wires):
    out = []
    for layer, (param, (left, right)) in list(enumerate(zip(params, wires)))[::-1]:
        # print(param.shape, left.shape, right.shape)
        instrs = ext_layer(param, left, right, layer)
        out = instrs + out
    root = f"g_{len(params)}_{0}"
    out = ext_elim(out, root)
    out = ext_copy_prop(out, root)
    out = ext_alpha_rename(out, root)
    return out

def ext_format(instr):
    (o, idx, l, r) = instr
    name = ext_gate_name(idx, l, r)
    return f"    cell {o} = {name};\n"

def ext_elim(instrs, root):
    out = []
    req = set([root])
    for instr in instrs[::-1]:
        (o, idx, l, r) = instr
        if o in req:
            req = ext_add_deps(req, idx, l, r)
            out.append(instr)
    return list(out[::-1])

def ext_copy_prop(instrs, root):
    out = []
    rename = dict()
    for instr in instrs:
        (o, idx, l, r) = instr
        if l in rename: l = rename[l]
        if r in rename: r = rename[r]
        if o == root: out.append((o, idx, l, r))
        elif idx == 3: rename[o] = l
        elif idx == 5: rename[o] = r
        else: out.append((o, idx, l, r))
    return out

def ext_alpha_count():
    # j and q, only letters not used in c keywords
    # no d or i to avoid do, if, in
    letters = "abcefghjklmnopqrstuvwxyz"
    for letter in letters:
        yield letter
    for letter in ext_alpha_count():
        for subletter in letters:
            yield letter + subletter

# for count in ext_alpha_count():
#     print(count)

def ext_regs_unique(instrs):
    seen = set()
    return [(o, seen.add(o))[0] for (o, _, _, _) in instrs if o not in seen]

def ext_alpha_rename(instrs, root):
    imm_regs = ext_regs_unique(instrs)
    rename = dict(zip(imm_regs, ext_alpha_count()))
    if root in rename: rename[root] = "out"
    for i in range(9): rename[f"g_0_{i}"] = f"in_{i}"
    out = []
    for (o, idx, l, r) in instrs:
        o = rename[o] if o in rename else o
        l = rename[l] if l in rename else l
        r = rename[r] if r in rename else r
        out.append((o, idx, l, r))
    return out

def ext_compile_to_c(params, wires):
    with open("gate.c.template", "r") as fin:
        before, after = fin.read().split("    {{ logic }}\n")
    with open("gate.c", "w") as fout:
        fout.write(before)
        for instr in ext_logic(params, wires):
            fout.write(ext_format(instr))
        fout.write(after)
    print("wrote circuit to gate.c")

if __name__ == "__main__":
    key = random.PRNGKey(379009)
    param_key, train_key = random.split(key)

    layer_sizes = [9, 64, 64, 64, 8, 4, 2, 1 ]

    params = rand_network(param_key, layer_sizes)

    params = train_adamw(train_key, params, epochs=60000 )

    ext_compile_to_c(params[0], zip(*params[1:]) )
