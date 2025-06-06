!******************************************************************************
!****m* /particleDefinition
! NAME
! module particleDefinition
! PURPOSE
! Here type(particle) is defined.
! This module includes also functions for this type.
!
! NOTES
! The type is enhanced by the additional field 'productionPos', which is only
! implemented, if the code is compiled with the flags
! -DWITHPRODUCTIONPOS=1.
! You never should access this field directly, only by the setter and getter
! routines provided here.
!******************************************************************************
Module particleDefinition

  implicit none
  private

  !****************************************************************************
  !****t* particleDefinition/particle
  ! NAME
  ! type particle
  ! PURPOSE
  ! This is the major type definition.
  !
  ! NOTES
  ! In order to calculate the size used in memory:
  ! * real   : 8 byte
  ! * integer: 4 byte
  ! * logical: 4 byte
  !
  ! This type definition has to be as small as possible.
  !
  ! If you want to store additional informations,
  ! you have to use "PIL" modules.
  !
  !
  ! SOURCE
  !
  Type particle
     sequence
     real, dimension (1:3) :: pos=0.
     real, dimension (0:3) :: mom=0.
     real, dimension (1:3) :: vel=0.           ! velocity=dr/dt in calc frame
     real                  :: mass=0.
     ! Note :
     ! This is not the invariant mass p^mu p_mu, but the bare mass of the particle without the self-energy shift:
     ! => p(0)      =   sqrt[mass+scalarPot)**2+p(1:3)**2]
     ! => p^mu p_mu =   (mass+scalarPot)**2
     real                  :: lastCollTime=0.  ! time of last collision
     real                  :: prodTime=0.      ! time of production
     real                  :: formTime=0.      ! time of formation
     real                  :: perWeight=0.     ! perturbative weight
     real                  :: scaleCS=1.       ! scaling factor for hadron cross section (during formation)
     real                  :: offshellPar=0.
     integer               :: ID=0
     integer               :: number=0         ! unique number for every particle
     integer               :: charge=0
     integer,dimension(1:2):: event=0          ! Number of event in which the particle was generated,
                                               ! changes during the run.
                                               ! cf. "collisionNumbering.f90" for details.
     integer               :: firstEvent=0     ! Number of first event, important for perturbative particles
                                               ! to track particle back to its production event.
                                               ! should stay constant during the run if the value is >0!
                                               ! is inherited to reaction products of the particle.
     integer               :: history=0        ! Variable to store the collision history of a particle
     logical               :: anti=.false.
     logical               :: pert=.false.
     logical               :: inF =.false.     ! ='in formation'

#ifdef WITHPRODUCTIONPOS
     real, dimension (1:3) :: prodPos=0.       ! position where particle was produced
#endif

  End Type particle
  !****************************************************************************

#ifdef WITHPRODUCTIONPOS
#warning "compiling with WITHPRODUCTIONPOS"
#endif

  !****************************************************************************
  !****f* particleDefinition/sqrtS
  ! NAME
  ! real function sqrtS(...)
  ! PURPOSE
  ! Evaluates Sqrt(s) for 1,2,3 or more particles,
  ! i.e.
  !    sqrt(mom(0)**2 - dot_product(mom(1:3),mom(1,3)))
  ! where "mom" is the sum of momenta of all given particles.
  !
  ! USAGE
  ! * (real)=sqrtS(x)
  ! * (real)=sqrtS(x,y)
  ! * (real)=sqrtS(x,y,z) with x,y,z of type particle
  ! * (real)=sqrtS(V) with V = vector of type particle
  !
  ! NOTES
  ! This function is overloaded.
  !
  ! You can also give an additional text as last argument:
  ! Then, before taking the sqrt, a test on negative argument is done.
  ! If the argument is smaller than -1e-3 an error message including the
  ! given text is thrown. (The threshold is given by the internal parameter
  ! "sqrtsCut".)
  !****************************************************************************
  Interface sqrtS
     Module Procedure sqrtSOne,sqrtSOneT,sqrtSTwo,sqrtSThree,sqrtSVec,sqrtSVecT !,sqrtSTwoT,sqrtSThreeT
  End Interface



  !****************************************************************************
  !****s* particleDefinition/setNumber
  ! NAME
  ! subroutine setNumber(teilchen)
  ! PURPOSE
  ! Set number of a particle, a vector of particles or an array of particles,
  ! which is unique.
  ! INPUTS
  ! * type(particle),intent(INout) :: teilchen
  !
  ! or:
  ! * type(particle),intent(INout),dimension(:) :: teilchen
  !
  ! or:
  ! * type(particle),intent(INout),dimension(:,:) :: teilchen
  !****************************************************************************
  Interface setNumber
     Module Procedure setNumber_0,setNumber_1,setNumber_2
  End Interface

  !****************************************************************************
  !****s* particleDefinition/setProductionPos
  ! NAME
  ! subroutine setProductionPos(p)
  ! INPUTS
  ! * type(particle) :: p -- the particle to be modified
  !
  ! or:
  ! * type(particle),dimension(:) :: p -- the particle to be modified
  !
  ! PURPOSE
  ! Set the additional entries ProductionPos to the value of position
  !
  ! Only has a function if compiled with -DWITHPRODUCTIONPOS=1
  !
  ! USAGE
  ! if (useProductionPos) call setProductionPos(pPart)
  !****************************************************************************
  Interface setProductionPos
     module procedure setProductionPos1, setProductionPos2
  end Interface setProductionPos

  !****************************************************************************

  type(particle), save :: p0  ! standard-initialized, used by 'setToDefault'

  !****************************************************************************

  integer, parameter :: number0 = 100000
  integer, save :: number = number0

  !****************************************************************************

  integer, parameter :: numberGuess0 = 100000
  integer, save :: numberGuess = numberGuess0
  logical, save :: lastNumberWasGuessed

  !****************************************************************************

  real,parameter :: sqrtsCut = -1e-3

  !****************************************************************************

  !****************************************************************************
  !****g* particleDefinition/useProductionPos
#ifdef WITHPRODUCTIONPOS
  logical, parameter :: useProductionPos = .true.
#else
  logical, parameter :: useProductionPos = .false.
#endif
  ! PURPOSE
  ! Parameter flag to indicate whether the code is compiled with the
  ! additional fields ProductionPos.
  !
  ! The compiler should avoid creating 'if'-blocks checking this flag,
  ! because it is stored as 'parameter', so there should be no size or time
  ! overhead.
  !****************************************************************************

  public :: particle
  public :: sqrtS
  public :: setToDefault
  public :: freeEnergy
  public :: KineticEnergy
  public :: absPos
  public :: absMom
  public :: rapidity
  public :: IsSamePart
  public :: getNumber
  public :: setNumber
  public :: setnumberguess
  public :: resetNumberGuess
  public :: AcceptGuessedNumbers
  public :: countParticles
  public :: setNumbersToDefault
  public :: useProductionPos
  public :: setProductionPos
  public :: getProductionPos

contains

  !****************************************************************************
  !****f* particleDefinition/FreeEnergy
  ! NAME
  ! real function FreeEnergy(x)
  ! PURPOSE
  ! Evaluates vacuum energy of particle (i.e. sqrt(mass^2+p^2) )
  ! INPUTS
  ! type(particle), intent(in) :: x
  ! USAGE
  ! (real)=FreeEnergy(x) with x of type(particle)
  !****************************************************************************
  pure real function FreeEnergy(x)
    type(particle), intent(in) :: x
    FreeEnergy=sqrt(x%mass**2+sum(x%mom(1:3)**2))
  end function FreeEnergy


  !****************************************************************************
  !****f* particleDefinition/kineticEnergy
  ! NAME
  ! real function kineticEnergy(x)
  ! PURPOSE
  ! Evaluates kinetic Energy= sqrt(p^2+m^2)-m for a particle
  ! INPUTS
  ! type(particle), intent(in) :: x
  ! USAGE
  ! (real)=kineticEnergy(x) with x of type particle
  ! RETURN VALUE
  ! real
  !****************************************************************************
  pure real function kineticEnergy(x)
    type(particle), intent(in) :: x
    kineticEnergy=sqrt(x%mass**2+sum(x%mom(1:3)**2))-x%mass
  end function kineticEnergy


  !****************************************************************************
  !****f* particleDefinition/absMom
  ! NAME
  ! real function absMom(x)
  ! PURPOSE
  ! Evaluates absolute momentum= SQRT(p,p) for a particle
  ! INPUTS
  ! type(particle), intent(in) :: x
  ! USAGE
  ! (real)=absMom(x) with x of type particle
  ! RETURN VALUE
  ! real
  !****************************************************************************
  pure real function absMom(x)
    type(particle), intent(in) :: x
    absMom=sqrt(sum(x%mom(1:3)**2))
  end function absMom


  !****************************************************************************
  !****f* particleDefinition/absPos
  ! NAME
  ! real function absPos(x)
  ! PURPOSE
  ! Evaluates absolute position= SQRT(x,x) for a particle
  ! INPUTS
  ! type(particle), intent(in) :: x
  ! USAGE
  ! (real)=absPos(x) with x of type particle
  ! RETURN VALUE
  ! real
  !****************************************************************************
  pure real function absPos(x)
    type(particle), intent(in) :: x
    absPos=sqrt(sum(x%pos(1:3)**2))
  end function absPos


  !****************************************************************************
  !****f* particleDefinition/rapidity
  ! NAME
  ! real function rapidity(x)
  ! PURPOSE
  ! Evaluates rapidity y for a particle
  ! INPUTS
  ! type(particle), intent(in) :: x
  ! USAGE
  ! (real)=rapidity(x) with x of type particle
  ! RETURN VALUE
  ! real
  !****************************************************************************
  pure real function rapidity(x)
    type(particle), intent(in) :: x
    rapidity = 0.5 * log((x%mom(0)+x%mom(3))/(x%mom(0)-x%mom(3)))
  end function rapidity


  !****************************************************************************
  ! cf. interface "sqrtS" :
  !****************************************************************************
  pure real function sqrtSOne(x) !Sqrt(s) of one particle
    type(particle), intent(in) :: x
    sqrtSOne=sqrt(x%mom(0)**2-sum(x%mom(1:3)**2))
  end function sqrtSOne

  !----------------------------------------------------------------------------
  real function sqrtSOneT(x,T) !Sqrt(s) of one particle
    type(particle), intent(in) :: x
    character(*),intent(in)  :: T
    real :: momentumSquare
    momentumSquare=x%mom(0)**2-sum(x%mom(1:3)**2)
    if (momentumSquare < sqrtSCut) then
       write(*,'(A,A)') 'ATTENTION: argument of sqrtS too negative: ',T
       write(*,'(1X,4e12.5)') x%mom
       write(*,'(1X,A,e12.5)') ' ---> ',momentumSquare
       sqrtSOneT = 0.0
    else
       sqrtSOneT=sqrt(max(0.0,momentumSquare))
    end if
  end function sqrtSOneT

  !----------------------------------------------------------------------------
  pure real function sqrtSTwo(x,y) !Sqrt(s) of two particles
    type(particle), intent(in) :: x,y
    real :: ptot(0:3)
    ptot = x%mom + y%mom
    sqrtSTwo=sqrt(ptot(0)**2-sum(ptot(1:3)**2))
  end function sqrtSTwo

  !----------------------------------------------------------------------------
!   real function sqrtSTwoT(x,y,T) !Sqrt(s) of two particles
!     type(particle), intent(in) :: x,y
!     character(*),intent(in)  :: T
!     real :: momentumSquare
!     real :: ptot(0:3)
!     ptot = x%mom + y%mom
!     momentumSquare=ptot(0)**2-sum(ptot(1:3)**2)
!     if (momentumSquare < sqrtSCut) then
!        write(*,'(A,A)') 'ATTENTION: argument of sqrtS too negative: ',T
!        write(*,'(1X,4e12.5)') x%mom
!        write(*,'(1X,4e12.5)') y%mom
!        write(*,'(1X,A,e12.5)') ' ---> ',momentumSquare
!        sqrtSTwoT = 0.0
!     else
!        sqrtSTwoT=sqrt(max(0.0,momentumSquare))
!     end if
!   end function sqrtSTwoT

  !----------------------------------------------------------------------------
  pure real function sqrtSThree(x,y,z) !Sqrt(s) of three particles
    type(particle), intent(in) :: x,y,z
    real :: ptot(0:3)
    ptot = x%mom + y%mom + z%mom
    sqrtSThree=sqrt(ptot(0)**2-sum(ptot(1:3)**2))
  end function sqrtSThree

  !----------------------------------------------------------------------------
!   real function sqrtSThreeT(x,y,z,T) !Sqrt(s) of three particles
!     type(particle), intent(in) :: x,y,z
!     character(*),intent(in)  :: T
!     real :: momentumSquare
!     real :: ptot(0:3)
!     ptot = x%mom + y%mom + z%mom
!     momentumSquare=ptot(0)**2-sum(ptot(1:3)**2)
!     if (momentumSquare < sqrtSCut) then
!        write(*,'(A,A)') 'ATTENTION: argument of sqrtS too negative: ',T
!        write(*,'(1X,4e12.5)') x%mom
!        write(*,'(1X,4e12.5)') y%mom
!        write(*,'(1X,4e12.5)') z%mom
!        write(*,'(1X,A,e12.5)') ' ---> ',momentumSquare
!        sqrtSThreeT = 0.0
!     else
!        sqrtSThreeT=sqrt(max(0.0,momentumSquare))
!     end if
!   end function sqrtSThreeT

  !----------------------------------------------------------------------------
  pure real function sqrtSVec(Part) ! Sqrt(s) of Vector
    type(particle), intent(in), dimension(:) :: Part
    integer :: i
    real :: ptot(0:3)
    ptot = 0.
    do i=1,size(Part,dim=1)
       ptot = ptot + Part(i)%mom
    end do
    sqrtSVec=sqrt(ptot(0)**2-sum(ptot(1:3)**2))
  end function sqrtSVec

  !----------------------------------------------------------------------------
  real function sqrtSVecT(Part,T) ! Sqrt(s) of Vector
    type(particle), intent(in), dimension(:) :: Part
    character(*),intent(in)  :: T
    integer :: i
    real :: ptot(0:3)
    real :: momentumSquare
    ptot = 0.
    do i=1,size(Part,dim=1)
       ptot = ptot + Part(i)%mom
    end do
    momentumSquare=ptot(0)**2-sum(ptot(1:3)**2)
    if (momentumSquare < sqrtSCut) then
       write(*,'(A,A)') 'ATTENTION: argument of sqrtSVec too negative: ',T
       write(*,'(1X,4e12.5)') ptot
       write(*,'(1X,A,e12.5)') ' ---> ',momentumSquare
       sqrtSVecT = 0.0
    else
       sqrtSVecT=sqrt(max(0.0,momentumSquare))
    end if

  end function sqrtSVecT


  !****************************************************************************
  !****s* particleDefinition/setToDefault
  ! NAME
  ! subroutine setToDefault(teilchen)
  ! PURPOSE
  ! Reset the particle to default values.
  ! INPUTS
  ! * type(particle) :: teilchen
  ! OUTPUT
  ! Returns "teilchen" with all structure elements set to the default values.
  ! NOTE
  ! This routine is elemental, i.e. it can be applied to scalars as well as
  ! arrays.
  !****************************************************************************
  elemental subroutine setToDefault(teilchen)
    type(particle),intent(inOut) :: teilchen
    teilchen = p0
  end subroutine setToDefault


  !****************************************************************************
  !****s* particleDefinition/setNumberGuess
  ! NAME
  ! subroutine setNumberGuess(teilchen)
  ! PURPOSE
  ! As the subroutine "setNumber", set the (unique) number of a particle.
  ! But here: Do this on a preliminary way.
  ! INPUTS
  ! * type(particle),intent(INout) :: teilchen
  ! RESULT
  ! teilchen%number is set
  ! NOTES
  ! During generation of high energetic events, the ("unique") number of
  ! a particle has already to be known before they are really inserted into
  ! the particle vector and the numbers are assigned.
  ! But some events are skipped an redone resulting in a new particle list.
  ! Therefore the former "SetNumberGuess" calls can be forgotten by a call
  ! to "resetNumberGuess" or can be kept via "AcceptGuessedNumbers"
  ! [A "Hurra" for the nomenclature ;)]
  !
  ! The Calls "SetNumberGuess(...); ...;AcepptGuessedNumbers" is equivalent
  ! to "SetNumber"
  !****************************************************************************
  subroutine setNumberGuess(teilchen)
    type(particle),intent(INout) :: teilchen
    teilchen%number = numberGuess
    numberGuess=numberGuess+1
    lastNumberWasGuessed = .TRUE.
!    write(*,*) 'SetNumberGuess:',teilchen%number,teilchen%ID
  end subroutine setNumberGuess


  !****************************************************************************
  !****s* particleDefinition/resetNumberGuess
  ! NAME
  ! subroutine resetNumberGuess(wasGuessed)
  ! PURPOSE
  ! forget all preliminary set "unique" numbers of particles
  !
  ! cf. "setNumberGuess"
  ! INPUTS
  ! * logical, OPTIONAL :: wasGuessed -- possible value for the Flag
  !   lastNumberWasGuessed
  !****************************************************************************
  subroutine resetNumberGuess(WasGuessed)
    logical, intent(in), optional :: WasGuessed
!    write(*,*) 'resetNumberGuess:',number,numberGuess
    numberGuess = number
    if (present(WasGuessed)) then
       lastNumberWasGuessed = WasGuessed
    else
       lastNumberWasGuessed = .FALSE.
    end if
  end subroutine resetNumberGuess


  !****************************************************************************
  !****s* particleDefinition/AcceptGuessedNumbers
  ! NAME
  ! subroutine AcceptGuessedNumbers()
  ! PURPOSE
  ! accept all preliminary set "unique" numbers of particles
  !
  ! cf. "setNumberGuess"
  ! INPUTS
  ! none
  !****************************************************************************
  logical function AcceptGuessedNumbers()
 !   write(*,*) 'AcceptGuessedNumbers:',number,numberGuess,lastNumberWasGuessed
    AcceptGuessedNumbers = lastNumberWasGuessed
    if (lastNumberWasGuessed) then
       number = numberGuess
       lastNumberWasGuessed = .FALSE.
    end if
  end function AcceptGuessedNumbers



  !****************************************************************************
  ! cf. interface "setNumber" :
  !****************************************************************************
  subroutine setNumber_0(teilchen)
    use CALLSTACK
    type(particle),intent(INout) :: teilchen

    if (teilchen%number > 0) then
       if (teilchen%number .ne. number) then
          write(*,*) '!!!! setNumber: number wrong !!!',&
               teilchen%number,number,numberGuess
          call TRACEBACK(user_exit_code=-1)
       end if
    end if

    teilchen%number=number
    number=number+1
    lastNumberWasGuessed = .FALSE.
!    write(*,*) 'SetNumber:',teilchen%number,teilchen%ID
  end subroutine setNumber_0

  !-------------------------------------------------------------------------
  subroutine setNumber_1(teilchen)
    type(particle),intent(INout),dimension(:) :: teilchen
    integer :: i
    do i=lbound(teilchen,dim=1),ubound(teilchen,dim=1)
       call setNumber_0(teilchen(i))
    end do
  end subroutine setNumber_1

  !-------------------------------------------------------------------------
  subroutine setNumber_2(teilchen)
    type(particle),intent(INout),dimension(:,:) :: teilchen
    integer :: i
    do i=lbound(teilchen,dim=1),ubound(teilchen,dim=1)
       call setNumber_1(teilchen(i,:))
    end do
  end subroutine setNumber_2


  !****************************************************************************
  !****f* particleDefinition/getNumber
  ! NAME
  ! integer function getNumber
  ! PURPOSE
  ! return the number, which would be given as the (unique) number of a particle
  ! in a next call of setNumber
  ! INPUTS
  ! module variable "number"
  !****************************************************************************
  pure integer function getNumber()
    getNumber = number
  end function getNumber


  !****************************************************************************
  !****s* particleDefinition/setNumbersToDefault
  ! NAME
  ! subroutine setNumbersToDefault
  ! PURPOSE
  ! Set particle number counters to default values. Needed to avoid overflows
  ! for large number of parallel ensembles and/or particles. Is called
  ! from initconfig at the beginning of every subsequent run.
  ! INPUTS
  !****************************************************************************
  subroutine setNumbersToDefault
    number=number0
    numberGuess=numberGuess0
  end subroutine setNumbersToDefault


  !****************************************************************************
  !****f* particleDefinition/IsSamePart
  ! NAME
  ! logical function IsSamePart(x,y)
  ! PURPOSE
  ! Compare two particles and return .TRUE., if:
  ! * ID,
  ! * Charge and
  ! * antiparticle flag
  ! are equal.
  ! INPUTS
  ! * type(particle), intent(in) :: x,y
  !****************************************************************************
  pure logical function IsSamePart(x,y)
    type(particle), intent(in) :: x,y
    IsSamePart = (x%ID==y%ID) &
         .and. (x%charge==y%charge) .and. (x%anti.eqv.y%anti)
  end function IsSamePart


  !****************************************************************************
  !****f* particleDefinition/countParticles
  ! NAME
  ! integer function countParticles(p,id)
  ! PURPOSE
  ! Counts how many particles with ID "id" are in a given particle vector "p".
  ! INPUTS
  ! * type(particle), dimension(:,:), intent(in) :: p -- particle vector
  ! * integer, intent(in) :: id                       -- ID to check for
  ! OUTPUT
  ! * count
  !****************************************************************************
  integer function countParticles(p,id)
    type(particle), dimension(:,:), intent(in) :: p
    integer, intent(in) :: id
    integer :: i,j
    countParticles=0
    do j=lbound(p,dim=2),ubound(p,dim=2)
       do i=lbound(p,dim=1),ubound(p,dim=1)
          if (p(i,j)%ID==id) countParticles=countParticles+1
       end do
    end do
  end function countParticles



  !****************************************************************************
  ! cf. interface setProductionPos
  !****************************************************************************
  subroutine setProductionPos1(p)

#ifdef WITHPRODUCTIONPOS
    type(particle), intent(inOut) :: p
    p%prodPos = p%pos
#else
    type(particle), intent(in) :: p
    ! no function
#endif
  end subroutine setProductionPos1
  !----------------------------------------------------------------------------
  subroutine setProductionPos2(p)

#ifdef WITHPRODUCTIONPOS
    type(particle), dimension(:), intent(inOut) :: p
    integer :: i
    do i=lbound(p,dim=1),ubound(p,dim=1)
       p(i)%prodPos = p(i)%pos
    end do
#else
    type(particle), dimension(:), intent(in) :: p
    ! no function
#endif
  end subroutine setProductionPos2

  !****************************************************************************
  !****f* particleDefinition/getProductionPos
  ! NAME
  ! function getProductionPos(p)
  ! INPUTS
  ! * type(particle) :: p -- the particle to be accessed
  ! RESULT
  ! real, dimension(1:3) -- the positions stored in p
  ! PURPOSE
  ! return the value stored in the filed ProductionPos, or a dummy value
  ! (-99.9,-99.9,-99.9) if unused
  !****************************************************************************
  function getProductionPos(p)
    type(particle), intent(in) :: p
    real, dimension(1:3) :: getProductionPos ! the return value

#ifdef WITHPRODUCTIONPOS
    getProductionPos = p%prodPos
#else
    real, dimension(1:3), parameter :: dummy = (/-99.9, -99.9, -99.9 /)
    getProductionPos = dummy
#endif

  end function getProductionPos



End Module particleDefinition
