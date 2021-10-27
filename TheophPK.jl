# set up environment
cd(@__DIR__)
using Pkg
Pkg.activate(".")

# load packages
using RDatasets, DataFrames, DataFramesMeta, Query, Chain, CategoricalArrays, CSV  # data wrangling
using Statistics        # statistics
using Plots, Gadfly, StatsPlots  # plotting
using DifferentialEquations, ModelingToolkit  # modeling and simulation
using DiffEqSensitivity, GlobalSensitivity  # sensitivity analysis
using DiffEqParamEstim, Optim, Distributions, GalacticOptim, BlackBoxOptim  # parameter estimation
using Turing, LinearAlgebra, MCMCChains  # Bayesian inference
using RCall  # call R from Julia
using Random  # other

# set seed to reproduce
Random.seed!(1234)

# load Theoph datasets
Theoph = dataset("datasets", "Theoph")

############################################
############# Data wrangling ###############
############################################

## Add and rename columns to the familiar NM-TRAN dataset structure

# R:
## data1 <- Theoph %>%
##              mutate(amt = 0.0,
##                     evid = 0) %>%
##              rename(ID = Subject,
##                     dv = Conc) 

# Julia:
## Query
data1 = Theoph |>
    @mutate(amt = 0.0,
            evid = 0) |>    
    @rename(:Subject => :ID,
            :Conc => :dv) |>
    DataFrame

# Julia:
## DataFrames:
data1 = copy(Theoph)
insertcols!(data1, :amt => 0.0, :evid => 0)
rename!(data1, :Subject => :ID, :Conc => :dv)

# Julia:
## DataFramesMeta
### commonly used functions compared to dplyr
#### @subset = filter
#### @select = select
#### @transform = mutate
data1 = @transform(Theoph, :amt = 0.0, :evid = 0)
rename!(data1, :Subject => :ID, :Conc => :dv)

# Julia:
## DataFramesMeta with Chain
data1 = @chain begin
    Theoph
    @transform(:amt = 0.0, :evid = 0)
    rename(:Subject => :ID, :Conc => :dv)
end

ids = string.([1:1:12;])
levels!(data1.ID,ids)

## data_dose <- data1 %>%
##              filter(Time = 0.0) %>%
##              mutate(evid = 1,
##                     amt = Dose * Wt,
##                     dv = 0.0)

data_dose = @chain begin
    data1
    @subset(:Time .== 0.0) 
    @transform(:evid = 1, :amt = :Dose .* :Wt, :dv = 0.0)
end

## data2 <- bind_rows(data_dose, data1) %>%
##              arrange(ID, Time, evid) 

data2 = @chain begin
    vcat(data_dose, data1)
    DataFramesMeta.@orderby(:ID, :Time, :evid)
end

############################################
####### Exploratory data analysis ##########
############################################

tmp_df = @subset(data1, :ID .== "3")
mean(tmp_df.dv)  # mean(tmp_df$dv)
median(tmp_df.dv)  # median(tmp_df$dv)
maximum(tmp_df.dv)  # max(tmp_df$dv)

describe(tmp_df)
describe(data1)  # summary(data1)

## Plots
Plots.plot(tmp_df.Time, tmp_df.dv)
Plots.scatter!(tmp_df.Time, tmp_df.dv)

## Gadfly
## ggplot(data1, aes(x=Time, y=dv, color=ID)) + geom_line()
Gadfly.plot(data1, x=:Time, y=:dv, color=:ID, Geom.line)
Gadfly.plot(data1, x=:Time, y=:dv, color=:ID, Geom.line, Scale.y_log10, Theme(background_color = "white"))
Gadfly.plot(data1, x=:Time, y=:dv, color=:ID, Geom.line, Geom.point, Scale.y_log10, Theme(background_color = "white"))

## ggplot2 with RCall
@rput data1

R"""
library(dplyr)
library(ggplot2)
​
data_r <- mutate(data1, ii = 0, addl = 0)

ggplot(data = data1, aes(x = Time, y=dv, color=ID)) +
    geom_point() +
    geom_line()
"""

@rget data_r

############################################
######## Modeling and simulation ###########
############################################

## Brief intro to compartmental modeling

## R: deSolve, RxODE, nlmixr, mrgsolve

## Standard approach ##

# define model
## in-place
function pk1cpt!(du, u, p, t)
    du[1] = -p[1]*u[1]
    du[2] = (p[1]*u[1]) / p[3] - (p[2]/p[3])*u[2]
end

#=
## out-of-place 
function pk1cpt(u, p, t)
    ddepot = -p[1]*u[1]
    dcent = (p[1]*u[1]) / p[3] - (p[2]/p[3])*u[2]
    return [ddepot, dcent]
end
=#

# set conditions
u0 = [319.365, 0.0]
p = [2.0,4.0,35.0]
tspan = (0.0, 25.0)

# define ODE problem and solve
prob = ODEProblem(pk1cpt!, u0, tspan, p)
sol = solve(prob, Tsit5())  # solver options https://diffeq.sciml.ai/stable/solvers/ode_solve/

# handling solution
Array(sol)
Array(sol)[2,:]
sol[2,:]
DataFrame(sol)

# plot
## Plots
Plots.plot(sol)
Plots.plot(sol, vars=[2])

## Gadfly
sol_df = DataFrame(sol)
rename!(sol_df, ["time","depot","cent"])
Gadfly.plot(sol_df, x=:time, y=:cent, Geom.line, 
    layer(tmp_df, x=:Time, y=:dv, Geom.point),
    Theme(background_color = "white"))

#####

## Using ModelingToolkit ; https://mtk.sciml.ai/stable/ and https://www.youtube.com/watch?v=HEVOgSLBzWA&t=7164s##
@parameters ka CL V
@variables t depot(t) cent(t)
D = Differential(t)

eqs = [D(depot) ~ -ka*depot,
       D(cent) ~ (ka*depot)/V - (CL/V)*cent]

@named sys = ODESystem(eqs)

# conditions
u0_sys = [depot => 319.365,
          cent => 0.0]

p_sys = [ka => 2.0,
         CL => 4.0,
         V => 35.0]

tspan_sys = (0.0,25.0)

prob_sys = ODEProblem(sys, u0_sys, tspan_sys, p_sys)
sol_sys = solve(prob_sys,Tsit5())

# plot
Plots.plot(sol_sys)
Plots.plot(sol_sys, vars=(cent), label="pred")
Plots.scatter!(tmp_df.Time, tmp_df.dv, label="obs")

############################################
######### Sensitivity analysis #############
############################################

## R: FME, mrgsim.sa, sensitivity

## local
prob_sens = ODEForwardSensitivityProblem(pk1cpt!, u0, tspan, p)
sol_sens = solve(prob_sens,Tsit5())

x,dp = extract_local_sensitivities(sol_sens)

Plots.plot(sol_sens.t, dp[1][2,:], label = "ka") 
Plots.plot!(sol_sens.t, dp[2][2,:], label = "CL")
Plots.plot!(sol_sens.t, dp[3][2,:], label = "V") 

## global
### create function that takes in parameters and returns endpoints for sensitivity
f_globsens = function(p)
    tmp_prob = remake(prob, p = p)
    tmp_sol = solve(tmp_prob, Tsit5())
    [maximum(tmp_sol[2,:])]
end

#### Morris 
m = GlobalSensitivity.gsa(f_globsens, Morris(total_num_trajectory=1000, num_trajectory=150),[[0.0,10.0],[0.0,10.0],[30.0,40.0]])
m.means
m.variances

#### Sobol
s = GlobalSensitivity.gsa(f_globsens, Sobol(), [[0.0,10.0],[0.0,10.0],[30.0,40.0]], N=1000)
s.ST
s.S1

Plots.bar(["ka","CL","V"], s.ST[1,:], title="Total Order Indices", legend=false)
Plots.hline!([0.05], linestyle=:dash)
Plots.bar(["ka","CL","V"], s.S1[1,:], title="First Order Indices", legend=false)
Plots.hline!([0.05], linestyle=:dash)

############################################
########## Parameter estimation ############
############################################

## R: nloptr, optim

## using DiffEqParamEstim ##

# optimize parameters for one subject 
data_optim = @subset(data1, :ID .== "3")

## least squares
cost_function_l2 = build_loss_objective(prob,Tsit5(),L2Loss(data_optim.Time,data_optim.dv), save_idxs = [2], maxiters=10000,verbose=false)
result_l2 = optimize(cost_function_l2, p)
p_optim_l2 = result_l2.minimizer

## weighted least squares
wts = 1 ./ data_optim.dv
cost_function_wtl2 = build_loss_objective(prob,Tsit5(),L2Loss(data_optim.Time[2:end], data_optim.dv[2:end], data_weight=wts[2:end]), save_idxs = [2], maxiters=10000,verbose=false)
result_wtl2 = optimize(cost_function_wtl2, p)
p_optim_wtl2 = result_wtl2.minimizer

## maximum likelihood
distributions = [truncated(Normal(data_optim.dv[i], 0.05*data_optim.dv[i]), 0.0, Inf) for i in 2:length(data_optim.Time)]
cost_function_mle = build_loss_objective(prob,Tsit5(), LogLikeLoss(data_optim.Time[2:end], distributions), save_idxs = [2], maxiters=10000, verbose=false)
result_mle = optimize(cost_function_mle, p)
p_optim_mle = result_mle.minimizer

## MAP Bayes
priors = [Uniform(0.0, 5.0), Uniform(0.0, 5.0), Uniform(10.0, 50.0)]
cost_function_map = build_loss_objective(prob, Tsit5(), L2Loss(data_optim.Time, data_optim.dv), priors=priors, save_idxs = [2], maxiters=10000, verbose=false)
result_map = optimize(cost_function_map, p)
p_optim_map = result_map.minimizer

### get pred with optimized parameters
pred = function(p_optim)
    prob_optim = remake(prob, p=p_optim)
    sol_optim = solve(prob_optim, Tsit5(), saveat=data_optim.Time)
    return sol_optim
end

## predictions based on optimized params
sol_optim_l2 = pred(p_optim_l2)
sol_optim_wtl2 = pred(p_optim_wtl2)
sol_optim_mle = pred(p_optim_mle)
sol_optim_map = pred(p_optim_map)

## plot results
Plots.scatter(data_optim.Time, data_optim.dv, label="data")
Plots.plot!(sol.t, sol[2,:], label="initial", line = (:dash))
Plots.plot!(sol_optim_l2.t, sol_optim_l2[2,:], label="L2")
Plots.plot!(sol_optim_wtl2.t, sol_optim_wtl2[2,:], label="weighted L2")
Plots.plot!(sol_optim_mle.t, sol_optim_mle[2,:], label="MLE")
Plots.plot!(sol_optim_map.t, sol_optim_map[2,:], label="MAP")

####

## using GalacticOptim ##

## loss function
function loss(p, u0)
    tmp_prob = remake(prob, p=p, u0=u0)
    tmp_sol = Array(solve(tmp_prob, Tsit5(), saveat=data_optim.Time))
    loss = sum(abs2, data_optim.dv .- tmp_sol[2,:])
    return loss
end

### derivative-free
prob_optim = GalacticOptim.OptimizationProblem(loss, p, u0)
p_optim_nm = solve(prob_optim, NelderMead())

### gradient-based
f_optim = OptimizationFunction(loss, GalacticOptim.AutoForwardDiff())
prob_optim = GalacticOptim.OptimizationProblem(f_optim, p, u0, lb = [0.0,0.0,0.0], ub = [10.0,10.0,50.0])
p_optim_bfgs = solve(prob_optim, Fminbox(BFGS()))

### global
prob_optim = GalacticOptim.OptimizationProblem(loss, p, u0, lb = [0.0,0.0,0.0], ub = [10.0,10.0,50.0])
p_optim_bbo = solve(prob_optim, BBO_adaptive_de_rand_1_bin_radiuslimited())  # Differential Evolution optimizer

## predictions based on optimized params
sol_optim_nm = pred(p_optim_nm)
sol_optim_bfgs = pred(p_optim_bfgs)
sol_optim_bbo = pred(p_optim_bbo)

## plot results
Plots.scatter(data_optim.Time, data_optim.dv, label="data")
Plots.plot!(sol.t, sol[2,:], label="initial", line = (:dash))
Plots.plot!(sol_optim_nm.t, sol_optim_nm[2,:], label="Nelder-Mead")
Plots.plot!(sol_optim_bfgs.t, sol_optim_bfgs[2,:], label="BFGS")
Plots.plot!(sol_optim_bbo.t, sol_optim_bbo[2,:], label="BBO")

# optimize parameters for population ; naive pooled approach
doses = data_dose.amt 

## loss function
function loss_pop(p, doses)
    ids = unique(data1.ID)
    losses = []
    for i in 1:length(ids)
        tmp_df = @subset(data1, :ID .== string.(i))
        tmp_prob = remake(prob, u0=[doses[i],0.0], p=p)
        tmp_sol = Array(solve(tmp_prob, Tsit5(), saveat=tmp_df.Time))
        tmp_loss = sum(abs2, tmp_df.dv .- tmp_sol[2,:])
        push!(losses, tmp_loss)
    end
    loss = sum(losses)
    return loss
end

prob_optim_pop = GalacticOptim.OptimizationProblem(loss_pop, p, doses)
p_optim_pop = solve(prob_optim_pop, NelderMead())

# compare to published population NONMEM theophylline PK model results https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-017-0427-0#Sec8:
## ka = 1.46 /h ; CL = 2.88 L/h ; V = 33.01 L

############################################
########## Population simulation ###########
############################################

# create problem function to pass different doses
function prob_func(prob,i,repeat)
    u0_tmp = [doses[i],0.0]
    remake(prob, u0=u0_tmp, p=p_optim_pop)
end

ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
@time ensemble_sol = solve(ensemble_prob, Tsit5(), EnsembleSerial(), trajectories=length(doses))  # serial
@time ensemble_sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories=length(doses))  # parallel - default

# plot
Plots.plot(ensemble_sol, vars=[2])
Plots.scatter!(data1.Time, data1.dv)

############################################
############ Bayesian inference ############
############################################

## R: rstan, cmdstanr

# individual
@model function fitPKInd(data, prob)
    # priors
    σ ~ truncated(Cauchy(0.0, 0.5), 0.0, 2.0)
    ka ~ LogNormal(log(2.0), 0.2)
    CL ~ LogNormal(log(4.0), 0.2)
    V ~ LogNormal(log(35.0), 0.2)

    p = [ka,CL,V]
    prob = remake(prob, p=p)
    predicted = Array(solve(prob, Tsit5(), saveat=data_optim.Time))[2,:]

    # likelihood
    for i = 1:length(predicted)
        data[i] ~ Normal(predicted[i], σ)
    end
end

model = fitPKInd(data_optim.dv, prob)

# This next command runs 3 independent chains without using multithreading.
#@time chain = sample(model, NUTS(250, .65), MCMCSerial(), 250, 4) 
@time chain = sample(model, NUTS(500, .8), MCMCThreads(), 500, 4)               # parallel
@time chain_prior = sample(model, Prior(), MCMCThreads(), 500, 4, progress=false)  # run chains with prior distributions

## get results
summ, quant = describe(chain)

## diagnostics
StatsPlots.plot(chain)

## predictive checks
data_missing = Vector{Missing}(missing, length(data_optim.dv)) # vector of `missing`
model_pred = fitPKInd(data_missing, prob)
pred = predict(model_pred, chain)  # posterior
pred_prior = predict(model_pred, chain_prior)

### summaries
summ_pred, quant_pred = describe(pred)
summ_pred_prior, quant_pred_prior = describe(pred_prior)

### plot
plot_posteriorpp = Gadfly.plot(x=data_optim.Time, y=data_optim.dv, Geom.point, Theme(background_color = "white"), Guide.xlabel("Time"), Guide.ylabel("Concentration"), Guide.title("Posterior predictive check"),
    layer(x=data_optim.Time, y=quant_pred[:,4], Geom.line),
    layer(x=data_optim.Time, ymin=quant_pred[:,2], ymax=quant_pred[:,6], Geom.ribbon))

plot_priorpp = Gadfly.plot(x=data_optim.Time, y=data_optim.dv, Geom.point, Theme(background_color = "white"), Guide.xlabel("Time"), Guide.ylabel("Concentration"), Guide.title("Prior predictive check"),
    layer(x=data_optim.Time, y=quant_pred_prior[:,4], Geom.line),
    layer(x=data_optim.Time, ymin=quant_pred_prior[:,2], ymax=quant_pred_prior[:,6], Geom.ribbon))

hstack(plot_priorpp, plot_posteriorpp)

###

## note: following section might take a couple of minutes to run

# population
times = [data1.Time[data1.ID .== string(i)] for i in 1:12]
doses = data_dose.amt
nSubject = 12
bws = data_dose.Wt

@model function fitPKPop(data, prob, nSubject, doses, times, bws)
    # priors
    ## residual error
    σ ~ truncated(Cauchy(0.0, 0.5), 0.0, 2.0)
    
    ## population params
    k̂a ~ LogNormal(log(2.0), 0.2)
    ĈL ~ LogNormal(log(4.0), 0.2)
    V̂ ~ LogNormal(log(35.0), 0.2)

    # IIV
    ωₖₐ ~ truncated(Cauchy(0.0, 0.5), 0.0, 2.0)

    CLᵢ = ĈL .* (bws ./ 70.0).^0.75
    Vᵢ = V̂ .* (bws ./ 70.0)

    # centered parameterization
    # kaᵢ ~ filldist(LogNormal(log(k̂a), ωₖₐ), nSubject)

    # non-centered parameterization
    ηᵢ ~ filldist(Normal(0.0, 1.0), nSubject)
    kaᵢ = k̂a .* exp.(ωₖₐ .* ηᵢ)

    function prob_func(prob,i,repeat)
        u0_tmp = [doses[i],0.0]
        ps = [kaᵢ[i], CLᵢ[i], Vᵢ[i]]
        remake(prob, u0=u0_tmp, p=ps, saveat=times[i])
    end

    tmp_ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
    tmp_ensemble_sol = solve(tmp_ensemble_prob, Tsit5(), trajectories=nSubject) 

    predicted = reduce(vcat, [Array(tmp_ensemble_sol[i])[2,:] for i in 1:nSubject])

    # likelihood
    for i = 1:length(predicted)
        data[i] ~ Normal(predicted[i], σ)
    end
end

model_pop = fitPKPop(data1.dv, prob, nSubject, doses, times, bws)

# This next command runs 3 independent chains without using multithreading.
#@time chain_pop = sample(model_pop, NUTS(250,.65), MCMCSerial(), 250, 4)  # parallel
@time chain_pop = sample(model_pop, NUTS(500,.8), MCMCThreads(), 500, 4)  # parallel

## save mcmcchains
#write("BayesPopChains.jls", chain_pop)

##load saved mcmcchains
#chain_pop = read("BayesPopChains.jls", Chains)

## get results
summ_pop, quant_pop = describe(chain_pop)

# save and load results
#CSV.write("BayesPopSumm.csv", DataFrame(summ_pop))  # save
#summ_load = CSV.read("BayesPopSumm.csv", DataFrame)

## diagnostics
plot_chains = StatsPlots.plot(chain_pop[:,1:5,:])
#savefig(plot_chains, "BayesPopChains.pdf")

############################################
################ Simulation ################
############################################

#--# scenario 1 #--#

## population simulation with single 600 mg dose

### define new problem for simulation
doses_sim = repeat([600.0], nSubject)
times_sim = [[0.0:0.1:24.0;] for i in 1:nSubject]
prob_sim = remake(prob, u0=[600.0,0.0], tspan=[0.0,24.0])

### run simulation
data_missing = Vector{Missing}(missing, length(times_sim[1])*nSubject) # vector of `missing`
model_sim = fitPKPop(data_missing, prob_sim, nSubject, doses_sim, times_sim, bws)
sim = predict(model_sim, chain_pop)  # posterior

### create sim DataFrame and get stats
df_sim = @chain begin
    DataFrame(sim)
    DataFramesMeta.stack(3:2894)
    DataFramesMeta.@orderby(:iteration, :chain)
    @transform(:time = repeat(reduce(vcat, times_sim), 2000))

    groupby([:iteration, :chain, :time])
    @transform(:lo = quantile(:value, 0.05),
               :med = quantile(:value, 0.5),
               :hi = quantile(:value, 0.95))
    
    groupby(:time)
    @transform(:loLo = quantile(:lo, 0.025),
               :medLo = quantile(:lo, 0.5),
               :hiLo = quantile(:lo, 0.975),
               :loMed = quantile(:med, 0.025),
               :medMed = quantile(:med, 0.5),
               :hiMed = quantile(:med, 0.975),
               :loHi = quantile(:hi, 0.025),
               :medHi = quantile(:hi, 0.5),
               :hiHi = quantile(:hi, 0.975))
end

df_sim_summ = DataFramesMeta.@orderby(unique(df_sim[:,[5;9:17]]), :time)

### plot
Gadfly.plot(x=df_sim_summ.time, ymin=df_sim_summ.loMed, ymax=df_sim_summ.hiMed, Geom.ribbon, Theme(default_color="deepskyblue", background_color="white"), alpha=[0.8], Guide.xlabel("Time"), Guide.ylabel("Concentration", orientation=:vertical), Guide.title("Simulation: Single 600 mg dose - population"),
    layer(x=df_sim_summ.time, ymin=df_sim_summ.loLo, ymax=df_sim_summ.hiLo, Geom.ribbon, Theme(default_color="deepskyblue"), alpha=[0.5]),
    layer(x=df_sim_summ.time, ymin=df_sim_summ.loHi, ymax=df_sim_summ.hiHi, Geom.ribbon, Theme(default_color="deepskyblue"), alpha=[0.5]),
    layer(x=df_sim_summ.time, y=df_sim_summ.medMed, Geom.line, Theme(default_color="black")),
    layer(x=df_sim_summ.time, y=df_sim_summ.medLo, Geom.line, Theme(default_color="black")),
    layer(x=df_sim_summ.time, y=df_sim_summ.medHi, Geom.line, Theme(default_color="black")))

#--# scenario 2 #--#

## mean subject simulation with multiple daily 300 mg doses
### extract mean subject parameters
p_optim_mean = summ_pop[2:4,2]
prob_sim = remake(prob, u0=[300,0.0], tspan=(0.0,144.0), p=p_optim_mean)

### set up callbacks
dosetimes = [0.0:24.0:5*24.0;]
affect!(integrator) = integrator.u[1] += 300.0
cb = PresetTimeCallback(dosetimes,affect!)

### simulate
sim = solve(prob_sim, Tsit5(), callback=cb, saveat=0.1)

### plot
Gadfly.plot(x=sim.t, y=sim[2,:], Geom.line, Theme(background_color = "white"), Guide.xlabel("Time"), Guide.ylabel("Concentration"), Guide.title("Simulation: Multiple daily 300 mg doses - mean individual"))