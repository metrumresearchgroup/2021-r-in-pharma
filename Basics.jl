# recommend watching https://www.youtube.com/watch?v=8h8rQyEpiZA&t=1765s

#--------------------# comments #-----------------#

# comment using hashtag

#=
multiple line comment
=#

# suppress output
3
3;

#--------------------# math #-----------------#

3 + 2
3 - 2
3 * 2
3 / 2
3 ^ 2
3 % 2

#--------------------# variables and types #-----------------#

a = 2
typeof(a)

a = 2.0
typeof(a)

a = convert(Int, a)

α = 3.0

# Excercises:

## Create a variable using a symbol or emoji - go crazy - and assign it an integer value

## Convert the varable to a float

#------------------# strings #----------------#

# differences from R: 
## single quotes in julia mean character not string
name = "ahmed "
name2 = """tim """

## concatenate
string(name, name2)
name * name2

## interpolate
println("my name is $name")
age = 20
println("I am $age years old")
println("I am $(age^2) years old")
println("I am ", age, " years old")

# Exercises:

## Print this statement: "My name is <name> and my age is <age>" replacing <anme> and <age> with variables carrying name and age info.

## Repeat the statement "I love julia, " 10 times

#------------------# data structures #----------------#

## tuples ; ordered but immutable
t = (4,5,6) 

## dictionaries ; unordered but mutable
d = Dict("a" => 1, "b" => 2, "c" => 3)
d["a"]
d["a"] = 4
d["d"] = 10

## arrays ; ordered and mutable
### differences from R:
#### need to broadcast to do elementwise operations in julia

v = [1,2,3]
v = zeros(3)
v = rand(3)
v = rand(1:5, 3)
v = [1,2,"hi"]
push!(v, "ahmed")
pop!(v)

m = [1 2 3; 4 5 6]
m = zeros(2,3)
m = rand(2,3)
m = rand(1:10, 2, 3)

r = rand(4,3,2)
r = [[1, 2, 3], [4, 5, 6]]

## indexing
### differences from R:
#### m[1,] in R but m[1,:] in julia

v[1]
v[1:end]
v[:]

m[2,1]
m[2:end]
m[1,:]

# Exercises:

## Try to multiply 3-element vectors v1 and v2 that carry some values. What happens?

## Create a 4x5 matrix and update the value in 3rd row, 2nd column

#------------------# loops #----------------#
### differences from R:
#### need to end with "end" in julia and no need for brackets ; same goes for condiionals and functions

## while loop
n = 0
while n < 10
    n = n + 1
    println(n)
end

## for loop
for i in 1:10
    println(i)
end

for i in 10:20; println(i); end

[i for i in 1:10]  # array comprehension

# Exercises:

## Print the statement "My name is <name>" while looping through a few names replacing <name>

## Create a 2D array that carries sums of numbers from 1 to 5
# 5×5 Matrix{Int64}:
# 2  3  4  5   6
# 3  4  5  6   7
# 4  5  6  7   8
# 5  6  7  8   9
# 6  7  8  9  10

#------------------# conditionals #----------------#

a = 3.0
if a < 5
    println("a is less than 5")
else
    println("a is larger than or equal 5")
end

if a < 5; println("a is less than 5"); else; println("a is larger than or equal 5"); end

a < 5 ? println("a is less than 5") : println("a is larger than 5")

# Exercises:

## Create an if statement to print a value if it's even and nothing if odd

#------------------# functions #----------------#

function f1(x)
    x*2
end

f2(x) = x*2

function f3(x)
    return x*2, x*3
end

# anonymous functions
x -> x^2
f4 = x -> x^2
map(x -> x^2, [1,2,3])

# multiple dispatch
g(x::String) = "my name is $x"
g(x::Int64) = x*3

#------------------# metaprogramming #----------------#

# macros
@time 1 + 2
@show +