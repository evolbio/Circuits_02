module Anomaly
include("Temporal.jl")
using .Temporal
include("MultiSensor.jl")
using .MultiSensor
include("Data.jl")
using .Data
export temporal, multisensor


end # module Anomaly
