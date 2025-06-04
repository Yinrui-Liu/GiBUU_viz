!******************************************************************************
!****m* /particlePointerList
! NAME
! module particlePointerList
!
! PURPOSE
! This module defines all necesary types for storing pointers to particles.
! This includes lists of particles and lists to lists and ...
!
! INPUTS
! (none)
!******************************************************************************
module particlePointerList

  use particleDefinition
  use particlePointerListDefinition

  implicit none

  private

  public :: PartList_Print
  public :: PartList_getPart
  public :: PartList_countPart
  public :: PartList_INIT
  public :: PartList_CLEAR
  public :: PartList_APPEND
  public :: PartList_PREPEND
!   public :: PartList_REMOVE

contains

  !****************************************************************************
  !****s* particlePointerList/PartList_Print
  ! NAME
  ! subroutine PartList_Print(L)
  !
  ! PURPOSE
  ! Loop over the List and print every particle
  !
  ! INPUTS
  ! * type(tParticleList) :: L -- The List
  !
  ! OUTPUT
  ! written on stdout
  !****************************************************************************
  subroutine PartList_Print(L)
    use output, only: WriteParticle

    type(tParticleList) :: L
    type(tParticleListNode), POINTER :: pNode

    write(*,*) "PartList_Print: ", L%nEntries

    pNode => L%first
    do
       if (.not. associated(pNode)) exit

       call WriteParticle(6,0,0,pNode%V)
!       write(*,*) pNode%V%A, pNode%V%B

       pNode=>pNode%next
    end do

  end subroutine PartList_Print


  !****************************************************************************
  !****s* particlePointerList/PartList_getPart
  ! NAME
  ! logical function PartList_getPart(L, n, P, ID, charge, anti, weightNonZero)
  !
  ! PURPOSE
  ! Loop over the List "L" and find the "n"-th particle in the list with
  ! %ID="ID", "%anti=antiparticle" and
  ! %charge="charge".  This particle "P" is then returned.
  !
  ! INPUTS
  ! * type(tParticleList) :: L -- The List of particles
  ! * integer, OPTIONAL :: ID     -- ID of particle which shall be returned
  ! * integer, OPTIONAL :: charge -- charge of particle which shall be returned
  ! * logical, OPTIONAL :: anti -- .false. if we search for a particle,
  !   .true. for an antiparticle
  ! * integer :: n      -- We return the n-th particle in the list with %ID=ID
  ! * logical, OPTIONAL :: weightNonZero -- if .true. only count particles with
  !   perweight > 0
  !
  ! OUTPUT
  ! * type(particle) :: P -- n-th particle with wished ID
  ! * logical        :: success -- True if it was possible to find n-Particles
  !   with the wished ID,charge,anti; False otherwise
  !****************************************************************************
  function PartList_getPart(L, n, P, ID, charge, anti, weightNonZero) result (success)

    type(tParticleList), intent(in) :: L
    integer, intent(in) :: n
    type(particle), intent(out) ::  P
    integer, intent(in), optional :: ID
    integer, intent(in), optional :: charge
    logical, intent(in), optional :: anti
    logical, intent(in), optional :: weightNonZero
    logical :: success

    type(tParticleListNode), POINTER :: pNode
    integer :: foundIDs
    logical :: checkWeight

    checkWeight =.false.
    if (present(weightNonZero)) checkWeight = weightNonZero

    ! Default return values
    success = .false.
    call SetToDefault(p)

    ! Search particle:
    pNode => L%first
    foundIDs = 0
    do
       if (.not. associated(pNode)) exit

       if (compare(pNode%V, ID, charge, anti)) then
          if (checkWeight) then
             if (pNode%V%perWeight > 1e-20) then
                foundIDs = foundIDs + 1
             end if
          else
             foundIDs = foundIDs + 1
          end if
          if (foundIDs == n) then
             ! n particles with the wanted ID are found, and the n-th particle is returned.
             p = pNode%V
             success = .true.
             return
          end if
       end if
       pNode => pNode%next
    end do

  end function PartList_getPart

  !****************************************************************************
  !****s* particlePointerList/PartList_countParticle
  ! NAME
  ! integer function PartList_countPart(L, ID, charge, anti, weightNonZero)
  !
  ! PURPOSE
  ! count the number of particles with given ID and/or charge and/or anti-flag
  !
  ! INPUTS
  ! * type(tParticleList) :: L -- The List of particles
  ! * integer :: ID     -- ID of particle which shall be returned
  ! * integer, OPTIONAL :: charge -- charge of particle which shall be returned
  ! * logical, OPTIONAL :: anti -- .false. if we search for a particle,
  !   .true. for an antiparticle
  ! * logical, OPTIONAL :: weightNonZero -- if .true. only count particles
  !   with perweight > 0
  !
  ! OUTPUT
  ! * integer
  !****************************************************************************
  function PartList_countPart(L, ID, charge, anti, weightNonZero) result(n)

    type(tParticleList), intent(in) :: L
    integer, intent(in), optional :: ID
    integer, intent(in), optional :: charge
    logical, intent(in), optional :: anti
    logical, intent(in), optional :: weightNonZero

    type(tParticleListNode), POINTER :: pNode
    integer :: n
    logical :: checkWeight

    checkWeight =.true.
    if (present(weightNonZero)) checkWeight = weightNonZero

    ! Default return values
    n = 0

    ! Search particle:
    pNode => L%first
    do
       if (.not. associated(pNode)) exit

       if (compare(pNode%V, ID, charge, anti)) n = n+1

       pNode => pNode%next
    end do

  end function PartList_countPart


  !****************************************************************************
  !****s* particlePointerList/PartList_INIT
  ! NAME
  ! subroutine PartList_INIT(L)
  !
  ! PURPOSE
  ! Initialize the List
  ! (call only at start; to reset the list please call PartList_CLEAR)
  !
  ! INPUTS
  ! * type(tParticleList) :: L -- The List
  !
  ! OUTPUT
  ! (none)
  !****************************************************************************
  subroutine PartList_INIT(L)

    type(tParticleList) :: L

    NULLIFY(L%first,L%last)
    L%nEntries=0

  end subroutine PartList_INIT



  !****************************************************************************
  !****s* particlePointerList/PartList_CLEAR
  ! NAME
  ! subroutine PartList_CLEAR(L,all)
  !
  ! PURPOSE
  ! Reset the List: Delete all Nodes and re-init the pointers
  !
  !
  ! INPUTS
  ! * type(tParticleList) :: L -- The List
  ! * logical, OPTIONAL   :: all -- if present and true, also the particle is deallocated
  !
  ! OUTPUT
  ! (none)
  !****************************************************************************
  subroutine PartList_CLEAR(L,all)

    type(tParticleList) :: L
    logical,optional :: all

    type(tParticleListNode), POINTER :: pNode,pNodeP

    if (L%nEntries>0) then
       pNodeP => L%first
       do
          if (.not. associated(pNodeP)) exit
          pNode => pNodeP%next
          if (present(all)) then
             if (all) deallocate(pNodeP%V)
          end if
          deallocate(pNodeP)
          pNodeP=>pNode
       end do
    end if
    call PartList_INIT(L)

  end subroutine PartList_CLEAR



  !****************************************************************************
  !****s* particlePointerList/PartList_APPEND
  ! NAME
  ! subroutine PartList_APPEND(L,V)
  !
  ! PURPOSE
  ! Append the particle (which V points at) at the end of the list.
  !
  !
  ! INPUTS
  ! * type(tParticleList)     :: L -- The List
  ! * type(particle), POINTER :: V -- The particle to add
  !
  ! OUTPUT
  ! (none)
  !****************************************************************************
  subroutine PartList_APPEND(L, V)

    type(tParticleList)              :: L
    type(particle)         , POINTER :: V

    type(tParticleListNode), POINTER :: pNode

    allocate(pNode)
    NULLIFY(pNode%next)
    pNode%V => V

    if (.not. associated(L%first)) then
       L%first => pNode
    else
       L%last%next => pNode
    end if
    L%last => pNode

    L%nEntries = L%nEntries+1

  end subroutine PartList_APPEND


  !****************************************************************************
  !****s* particlePointerList/PartList_PREPEND
  ! NAME
  ! subroutine PartList_PREPEND(L,V)
  !
  ! PURPOSE
  ! Prepend the particle (which V points at) at the beginning of the list.
  !
  !
  ! INPUTS
  ! * type(tParticleList)     :: L -- The List
  ! * type(particle), POINTER :: V -- The particle to add
  !
  ! OUTPUT
  ! (none)
  !****************************************************************************
  subroutine PartList_PREPEND(L, V)

    type(tParticleList)              :: L
    type(particle)         , POINTER :: V

    type(tParticleListNode), POINTER :: pNode

    allocate(pNode)
    NULLIFY(pNode%next)
    pNode%V => V

    if (.not. associated(L%last)) then
       L%last => pNode
    else
       pNode%next => L%first
    end if
    L%first => pNode

    L%nEntries = L%nEntries+1

  end subroutine PartList_PREPEND

  !****************************************************************************
  !****s* particlePointerList/PartList_RANDOMIZE
  ! NAME
  ! subroutine PartList_RANDOMIZE(L)
  !
  ! PURPOSE
  ! Reorder the linked list by reshuffling the entries randomly.
  !
  ! INPUTS
  ! * type(tParticleList)     :: L -- The List
  !
  ! OUTPUT
  ! (none)
  !****************************************************************************
  subroutine PartList_RANDOMIZE(L)

    use random, only: rn_truefalse

    type(tParticleList)              :: L


    type(tParticleListNode), POINTER :: pNode, pNodeNext

    if (L%nEntries < 2) return ! nothing to be reordered


    !
    ! NOT YET TESTED !!!!!!!!!!!!!!!!!!!!!!
    !

    pNode => L%first
    L%last => L%first
    pNodeNext => pNode%next
    NULLIFY(pNode%next)

    do while (associated(pNodeNext))
       pNode => pNodeNext
       pNodeNext => pNode%next

       if (rn_truefalse()) then ! PREPEND
          pNode%next => L%first
          L%first => pNode
       else ! APPEND
          L%last%next => pNode
          L%last => pNode
          NULLIFY(pNode%next)
       end if

    end do

  end subroutine PartList_RANDOMIZE


!  !****************************************************************************
!  !****f* particlePointerList/PartList_REMOVE
!  ! NAME
!  ! logical function PartList_REMOVE(L,iEntry,V)
!  !
!  ! PURPOSE
!  ! Remove the iEntry-th particle from the List.
!  ! The pointer to this particle is returned, the memory of the node is
!  ! freed.
!  !
!  ! NOTES
!  ! This routine is fastest if one removes the first particle. For all
!  ! higher indices, the routine has to loop over the entries of the list
!  ! until it has found the right number.
!  !
!  ! INPUTS
!  ! * type(tParticleList)     :: L -- The List
!  ! * integer                 :: iEntry -- The number of the particle to
!  !   remove
!  ! * type(particle), POINTER :: V -- The particle to add
!  !
!  ! OUTPUT
!  ! * type(particle), POINTER :: V -- The particle removed
!  ! * return value -- TRUE at success (index was in allowed range)
!  !****************************************************************************
!   logical function PartList_REMOVE(L,iEntry, V)
!
!     type(tParticleList)              :: L
!     type(particle)         , POINTER :: V
!     integer                          :: iEntry
!
!     type(tParticleListNode), POINTER :: pNode,pNodeP
!     integer :: i
!
!     NULLIFY(V)
!     PartList_REMOVE = .FALSE.
!
!     if (iEntry < 1) return
!     if (iEntry > L%nEntries) return
!
!     pNode => L%first
!     i = 2
!
!     if (L%nEntries == 1) then
!        V => pNode%V
!        DEALLOCATE(pNode)
!        NULLIFY(L%first,L%last)
!        L%nEntries = 0
!        PartList_REMOVE = .TRUE.
!        return
!     end if
!
!     if (iEntry == 1) then
!        V => pNode%V
!        L%first => pNode%next
!        DEALLOCATE(pNode)
!        L%nEntries = L%nEntries-1
!        PartList_REMOVE = .TRUE.
!        return
!     end if
!
!     do
!        if (i == iEntry) exit ! now pNode point to the precessor
!        i = i+1
!        pNode => pNode%next
!     end do
!
!     pNodeP => pNode
!     pNode => pNode%next
!
!     V => pNode%V
!     pNodeP%next => pNode%next
!
!     DEALLOCATE(pNode)
!     if (iEntry == L%nEntries) L%last => pNodeP
!     L%nEntries = L%nEntries-1
!     PartList_REMOVE = .TRUE.
!     return
!
!   end function PartList_REMOVE

  !****************************************************************************
  !****if* particlePointerList/compare
  ! NAME
  ! logical function compare(P, ID,charge,anti)
  !
  ! INPUTS
  ! * type(particle) :: P
  ! * integer, OPTIONAL :: ID
  ! * integer, OPTIONAL :: charge
  ! * logical, OPTIONAL :: anti
  !
  ! PURPOSE
  ! check whether the given particle has ID and/or charge and/or antiparticle
  ! flag.
  !****************************************************************************
  pure logical function compare(P, ID,charge,anti)
    type(particle), intent(in) :: P
    integer, intent(in), OPTIONAL :: ID
    integer, intent(in), OPTIONAL :: charge
    logical, intent(in), OPTIONAL :: anti

    compare = .false.
    if (present(ID)) then
       if (P%ID /= ID) return
    end if
    if (present(charge)) then
       if (P%charge /= charge) return
    end if
    if (present(anti)) then
       if (P%anti .neqv. anti) return
    end if
    compare = .true.

  end function compare

end module particlePointerList
