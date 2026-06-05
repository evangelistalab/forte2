def subspaces(*orbital_subspaces, rdm_orbitals=None):
    frag_orbs = []
    seen = set() #creates an unorded list of obritals stored in a single variable
     
    for subspace in orbital_subspaces:
        for orb in subspace:
            if orb in seen: #checks if the same orbital has already been added to the unordered list, meaning that orbital is redundant
                raise ValueError(f"Orbital {orb} appears in more than one subspace")
            seen.add(orb)
            frag_orbs.append(orb)

    local_pos = {orb: i for i, orb in enumerate(frag_orbs)}
    local_subspaces = [
        [local_pos[orb] for orb in subspace]
        for subspace in orbital_subspaces
    ] #indexing convention for the two-electron integral V

    if rdm_orbitals is None:
        global_subspaces = [list(subspace) for subspace in orbital_subspaces]
    else:
        rdm_pos = {orb: i for i, orb in enumerate(rdm_orbitals)}
        missing = sorted(set(frag_orbs) - set(rdm_pos))
        if missing:
            raise ValueError(
                f"Fragment orbitals are not present in the RDM orbital space: {missing}"
            )

        global_subspaces = [
            [rdm_pos[orb] for orb in subspace]
            for subspace in orbital_subspaces
        ] #indexing convention for the 2-rdm

    #Example: for a problem with core orbitals [0,1], the fragment active space might be [2,3,4]
    #V wants indices [2,3,4] while 2rdm is built for indices [0,1,2] corresponding to orbitals no. 2, 3, and 4

    return frag_orbs, global_subspaces, local_subspaces

