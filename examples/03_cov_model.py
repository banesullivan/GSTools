from gstools import CovModel
import numpy as np

# use CovModel as the base-class
class Gau(CovModel):
    def cor(self, h):
        return np.exp(-h ** 2)


model = Gau(dim=2, var=2.0, len_scale=10)

model.plot()

print(model.dim, model.var, model.len_scale, model.nugget, model.sill)
model.dim = 3
model.var = 1
model.len_scale = 15
model.nugget = 0.1
print(model.dim, model.var, model.len_scale, model.nugget, model.sill)

model = Gau(dim=3, var=2.0, len_scale=10, anis=0.5)
print(model.anis)
print(model.len_scale_vec)

model = Gau(dim=3, var=2.0, len_scale=[10, 5, 4])
print(model.anis)
print(model.len_scale)
print(model.len_scale_vec)

model = Gau(dim=3, var=2.0, len_scale=10, angles=2.5)
print(model.angles)

model = Gau(dim=3, var=2.0, len_scale=10, nugget=0.5)
print(model.variogram(10.0))
print(model.covariance(10.0))
print(model.correlation(10.0))

model = Gau(dim=3, var=2.0, len_scale=10)
print(model.spectrum(0.1))
print(model.spectral_density(0.1))

model = Gau(dim=3, var=2.0, len_scale=10)
print(model.integral_scale)
print(model.integral_scale_vec)

model = Gau(dim=3, var=2.0, integral_scale=[10, 4, 2])
print(model.anis)
print(model.len_scale)
print(model.len_scale_vec)
print(model.integral_scale)
print(model.integral_scale_vec)

model = Gau(dim=3, var=2.0, len_scale=10)
print(model.percentile_scale(0.9))


class Stab(CovModel):
    def default_opt_arg(self):
        return {"alpha": 1.5}

    def cor(self, h):
        return np.exp(-h ** self.alpha)


model1 = Stab(dim=2, var=2.0, len_scale=10)
model2 = Stab(dim=2, var=2.0, len_scale=10, alpha=0.5)
print(model1)
print(model2)

# data
x = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
y = [0.2, 0.5, 0.6, 0.8, 0.8, 0.9]
# fitting model
model = Stab(dim=2)
# we have to provide boundaries for the parameters
model.set_arg_bounds(alpha=[0, 3])
results, pcov = model.fit_variogram(x, y, nugget=False)
print(results)

ax = model.plot()
ax.scatter(x, y, color="k", label="estimated variogram")
ax.legend()
