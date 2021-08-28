function reverse_dims(A::AbstractArray)
    n = length(size(A))
    permutedims(A, n:-1:1)
end
