import docplex.mp.model as cpx
import pulp

mdl = cpx.Model()
x = mdl.binary_var(name="x")
mdl.minimize(x)
mps = mdl.export_as_mps_string()
with open("test.mps", "w") as f:
    f.write(mps)

vars_dict, prob = pulp.LpProblem.fromMPS("test.mps")
print(type(vars_dict), type(prob))
print("Dict keys:", vars_dict.keys())
status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
print("Status:", pulp.LpStatus[status])
for name, v in vars_dict.items():
    print(name, pulp.value(v))
