
using ACEpotentials, ACE1x, Distributed, DelimitedFiles

addprocs(8, exeflags="--project=$(Base.active_project())")

dimers = read_extxyz("./datasets/Si2.xyz")
trimers = read_extxyz("./datasets/Si3.xyz")
tetramers = read_extxyz("./datasets/Si4.xyz")
pentamers = read_extxyz("./datasets/Si5.xyz")

sym = [:Si]
rc = 5.5
nu = 4
totdeg = [24, 20, 16, 12]

dirty_basis = ACE1x.ace_basis(
   elements=sym, rcut=rc, order=nu,
   totaldegree=totdeg, pure=false, pure2b=false, delete2b=false,
);                        

# Compute non-orthogonalized ACE features
for (i, atoms) in enumerate(dimers)
   dirty_feat = site_descriptors(dirty_basis, atoms)
   open("./Si_ACE_featvecs/dirty/2mer/$(i-1).txt", "w") do cur_file
   writedlm(cur_file, reduce(hcat, dirty_feat)')
   end
end

for (i, atoms) in enumerate(trimers)
   dirty_feat = site_descriptors(dirty_basis, atoms)
   open("./Si_ACE_featvecs/dirty/3mer/$(i-1).txt", "w") do cur_file
   writedlm(cur_file, reduce(hcat, dirty_feat)')
   end
end

for (i, atoms) in enumerate(tetramers)
   dirty_feat = site_descriptors(dirty_basis, atoms)
   open("./Si_ACE_featvecs/dirty/4mer/$(i-1).txt", "w") do cur_file
   writedlm(cur_file, reduce(hcat, dirty_feat)')
   end
end

for (i, atoms) in enumerate(pentamers)
   dirty_feat = site_descriptors(dirty_basis, atoms)
   open("./Si_ACE_featvecs/dirty/5mer/$(i-1).txt", "w") do cur_file
   writedlm(cur_file, reduce(hcat, dirty_feat)')
   end
end

pure_basis = ACE1x.ace_basis(
   elements=sym, rcut=rc, order=nu,
   totaldegree=totdeg, pure=true, pure2b=false, delete2b=false,
);                        


# Compute purified/orthogonalized ACE features
for (i, atoms) in enumerate(dimers)
   pure_feat = site_descriptors(pure_basis, atoms)
   open("./Si_ACE_featvecs/pure/2mer/$(i-1).txt", "w") do cur_file
   writedlm(cur_file, reduce(hcat, pure_feat)')
   end
end

for (i, atoms) in enumerate(trimers)
   pure_feat = site_descriptors(pure_basis, atoms)
   open("./Si_ACE_featvecs/pure/3mer/$(i-1).txt", "w") do cur_file
   writedlm(cur_file, reduce(hcat, pure_feat)')
   end
end

for (i, atoms) in enumerate(tetramers)
   pure_feat = site_descriptors(pure_basis, atoms)
   open("./Si_ACE_featvecs/pure/4mer/$(i-1).txt", "w") do cur_file
   writedlm(cur_file, reduce(hcat, pure_feat)')
   end
end

for (i, atoms) in enumerate(pentamers)
   pure_feat = site_descriptors(pure_basis, atoms)
   open("./Si_ACE_featvecs/pure/5mer/$(i-1).txt", "w") do cur_file
   writedlm(cur_file, reduce(hcat, pure_feat)')
   end
end
