# load julia packages
using RCall, DifferentialEquations

# load R packages
R"""
library(dplyr)
library(ggplot2)
"""

## Do data wrangling in r using dplyr tools
R"""
data <- Theoph %>% 
    filter(Subject == 3) %>%
    mutate(amt = Dose*Wt)
"""

# export data from r to julia
@rget data

# define model and simulate in julia
function pk1cpt!(du, u, p, t)
    du[1] = -p[1]*u[1]
    du[2] = (p[1]*u[1]) / p[3] - (p[2]/p[3])*u[2]
end

# set conditions
dose = data.amt[data.Time .== 0.0][1]
u0 = [dose, 0.0]
p = [2.45, 2.89, 34.25]
tspan = (0.0, 25.0)

# define ODE problem and solve
prob = ODEProblem(pk1cpt!, u0, tspan, p)
pred = Array(solve(prob, Tsit5(), saveat=data.Time))[2,:]  # solver options https://diffeq.sciml.ai/stable/solvers/ode_solve/
​
# plot results in R
@rput pred
​
R"""
df <- tibble(time = data$Time,
             obs = data$conc,
             pred = pred)
​
ggplot(data = df, aes(x = time)) +
    geom_point(aes(y = obs)) +
    geom_line(aes(y = pred))
"""