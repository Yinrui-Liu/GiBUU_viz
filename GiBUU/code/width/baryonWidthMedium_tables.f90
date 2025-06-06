!******************************************************************************
!****m* /baryonWidthMedium_tables
! NAME
! module baryonWidthMedium_tables
! PURPOSE
! Implements the routines for the medium width of the baryons
!
! NOTES
! The corresponding tables are created by inMediumWidth.f90
!******************************************************************************
module baryonWidthMedium_tables

  use idTable, only: nres
  use CALLSTACK, only: TRACEBACK

  implicit none
  private

  public :: get_inMediumWidth
  public :: minimalWidth
  public :: dumpTable

  ! Grid parameters:
  public :: numSteps_rhoN
  public :: numSteps_rhoP
  public :: numSteps_absP
  public :: numSteps_mass
  public :: max_rhoN
  public :: max_rhoP
  public :: max_absP
  public :: inMediumParameterset

  public :: get_min_charge_loop,get_max_charge_loop
  public :: get_deltaOset
  public :: get_minMass
  public :: cleanUp

  !****************************************************************************
  !****g* baryonWidthMedium_tables/debugFlag
  ! SOURCE
  !
  logical, parameter :: debugFlag=.false.
  !
  ! PURPOSE
  ! Switch for debug information
  !****************************************************************************


  !****************************************************************************
  !****g* baryonWidthMedium_tables/maxRes
  ! SOURCE
  !
  integer,save  :: maxRes=1000
  !
  ! PURPOSE
  ! Read the data table up to a maximum resonance ID.
  ! ONLY FOR TESTING!!!
  !****************************************************************************

  !****************************************************************************
  !****g* baryonWidthMedium_tables/minRes
  ! SOURCE
  !
  integer,save  :: minRes=-1000
  !
  ! PURPOSE
  ! Read the data table starting at this minimal resonance ID.
  ! ONLY FOR TESTING!!!
  !****************************************************************************


  !****************************************************************************
  !****g* baryonWidthMedium_tables/deltaOset
  ! SOURCE
  !
  logical,save  :: deltaOset=.false.
  !
  ! PURPOSE
  ! Use delta width according to Oset et al. NPA 468 (1987)
  !****************************************************************************


  !****************************************************************************
  !****g* baryonWidthMedium_tables/minimalWidth
  ! SOURCE
  !
  real,parameter  :: minimalWidth=0.001
  !
  ! PURPOSE
  ! The minimal width a particle can get. Must be finite for numerics.
  !****************************************************************************

  !****************************************************************************
  !****g* baryonWidthMedium_tables/inMediumParameterset
  ! SOURCE
  !
  integer,save  :: inMediumParameterset=2
  !
  ! PURPOSE
  ! chooses the parameters for the inMediumWidth:
  ! * 1: charges accessible with electron
  ! * 2: charges accessible with neutrino (= all)
  !****************************************************************************


  !****************************************************************************
  !****g* baryonWidthMedium_tables/onlyNucleon
  ! SOURCE
  !
  logical, save :: onlyNucleon=.false.
  !
  ! PURPOSE
  ! Only for debugging: only nucleon width is read in.
  !****************************************************************************

  !****************************************************************************
  !****g* baryonWidthMedium_tables/extrapolateAbsP
  ! SOURCE
  !
  logical,save  :: extrapolateAbsP=.false.
  !
  ! PURPOSE
  ! if true then set absP to maxAbsP if absP is larger
  !****************************************************************************

  !****************************************************************************
  !****g* baryonWidthMedium_tables/FileNameMask
  ! PURPOSE
  ! The absolute filename of the file containing the in-medium tables.
  !
  ! This is only the basename, the final name is given by
  !   '[FileNameMask].iii_n_s.dat.bz2'
  ! where iii is the ID of the particle, n=1,2, and s=1,2
  !
  ! possible values:
  ! * if not set, default is '[path_To_Input]/inMediumWidth/InMediumWidth'
  ! * if given, but does not contain '/':
  !   default is '[path_To_Input]/inMediumWidth/[FileNameMask]'
  ! * otherwise: filename is absolute, including path
  !
  ! NOTE
  ! if you want to use the files 'XXX.iii_n_s.dat.bz2' in the actual directory,
  ! give this mask as './XXX'
  !
  ! SOURCE
  !
  character(1000), save :: FileNameMask = ''
  !****************************************************************************


  ! Grid parameters:
  integer, parameter:: numSteps_rhoN=4
  integer, parameter:: numSteps_rhoP=4
  integer, parameter:: numSteps_absP=75
  integer, parameter:: numSteps_mass=400
  real, parameter   :: max_rhoN=0.1*0.197**3 ! In GeV**3
  real, parameter   :: max_rhoP=0.1*0.197**3 ! In GeV**3
  real, parameter   :: max_absP=3.0

  logical, parameter :: fullInterpolation=.true.  !must be .true. !!!
  ! change only for debugging, otherwise problems with offshell gradients
  ! in propagation.f90

  logical,save  :: readInput_flag=.true.

  real(4), save, dimension(:,:,:,:,:,:), Allocatable:: widthTable ! ~ 180 MB total
  ! (:,:,:,:,:,1) = collisional broadening
  ! (:,:,:,:,:,2) = pauli blocked vacuum width


  real, save ::  delta_rhoN
  real, save ::  delta_rhoP
  real, save ::  delta_absP
  real, save, dimension(1:nres+1) :: delta_mass
  real, save, dimension(1:nres+1) :: min_mass
  real, save, dimension(1:nres+1) :: max_mass




contains

  !****************************************************************************
  !****************************************************************************
  real function get_minMass(id)
    use idtable, only: nucleon
    use particleProperties, only: hadron
    integer, intent(in) :: id

    if (readInput_flag) call readinput

    if (ID.eq.nucleon) then
       get_minMass=0.800 ! TODO: should be 0.700 or less
    else
       get_minMass=hadron(ID)%minmass+0.01
    end if
  end function get_minMass

  !****************************************************************************
  !****************************************************************************
  integer function get_min_charge_loop()

    if (readInput_flag) call readinput

    select case (inMediumParameterset)
    case (1)
       get_min_charge_loop= 0        ! Electron
    case (2)
       get_min_charge_loop=-1        ! Neutrino
    case default
       call TRACEBACK('error in get_min_charge_loop')
    end select
  end function get_min_charge_loop

  !****************************************************************************
  !****************************************************************************
  integer function get_max_charge_loop()

    if (readInput_flag) call readinput

    select case (inMediumParameterset)
    case (1)
       get_max_charge_loop=1       ! Electron
    case (2)
       get_max_charge_loop=2       ! Neutrino
    case default
       call TRACEBACK('error in get_max_charge_loop')
    end select
  end function get_max_charge_loop

  !****************************************************************************
  !****************************************************************************
  logical function get_deltaOset()

    if (readInput_flag) call readinput

    get_deltaOset=deltaOset
  end function get_deltaOset

  !****************************************************************************
  !****s* baryonWidthMedium_tables/readInput
  ! NAME
  ! subroutine readInput
  ! PURPOSE
  ! Reads input in jobcard from namelist "BaryonWidthMedium_tables".
  !****************************************************************************
  subroutine readInput
    use inputGeneral, only: path_To_Input, ExpandPath
    use output

    integer :: ios ! checks file behavior

    !**************************************************************************
    !****n*  baryonWidthMedium_tables/BaryonWidthMedium_tables
    ! NAME
    ! NAMELIST /BaryonWidthMedium_tables/
    ! PURPOSE
    ! Includes the input switches:
    ! * minRes
    ! * maxRes
    ! * inMediumParameterset
    ! * onlyNucleon
    ! * deltaOset
    ! * extrapolateAbsP
    ! * FileNameMask
    !**************************************************************************
    NAMELIST /baryonWidthMedium_tables/ minRes,maxRes,inMediumParameterset, &
         onlyNucleon, deltaOset, extrapolateAbsP, &
         FileNameMask

    if (.not.readInput_flag) return

    call Write_ReadingInput('baryonWidthMedium_tables',0)
    rewind(5)
    read(5,nml=baryonWidthMedium_tables,IOSTAT=IOS)
    call Write_ReadingInput('baryonWidthMedium_tables',0,IOS)

    write(*,*) "minRes = ", minRes
    write(*,*) "maxRes = ", maxRes
    write(*,*) "inMediumParameterset = ", inMediumParameterset, " (1:e, 2:nu)"
    write(*,*) "Oset Delta           = ", deltaOset
    write(*,*) "extrapolateAbsP      = ", extrapolateAbsP

    if (len_trim(FileNameMask)>0) then
       if (index(FileNameMask,"/")>0) then
          call ExpandPath(FileNameMask)
          FileNameMask = trim(FileNameMask)
       else
          FileNameMask = trim(path_to_Input)//'/inMediumWidth/'//trim(FileNameMask)
      end if
    else
       FileNameMask = trim(path_to_Input)//'/inMediumWidth/InMediumWidth'
    end if
    write(*,*) "filename mask: ",trim(FileNameMask)

    if ((inMediumParameterset < 1).or.(inMediumParameterset > 2)) then
       call TRACEBACK("wrong value for inMediumParameterset")
    end if

    readInput_flag=.false.

    call Write_ReadingInput('baryonWidthMedium_tables',1)

  end subroutine readInput

  !****************************************************************************
  !****************************************************************************
  subroutine cleanUp
    if (allocated(widthTable)) deallocate(widthTable)
  end subroutine


  !****************************************************************************
  !****f* baryonWidthMedium_tables/get_inMediumWidth
  ! NAME
  ! function get_inMediumWidth(particleID,momentum,baremass,rhoN,rhoP,switch,
  ! outOfBounds) result(width)
  !
  ! PURPOSE
  ! * Returns the in-medium width of baryons according to
  !   Gamma=Gamma_free*Pauli_Blocking+sigma*rho*v *Pauli_Blocking
  ! * An average over the Fermi see is performed
  ! * Sigma is given by the actual full collision term, with the incoming
  !   particles in the vacuum
  ! * Simplification: An average over the charge of the baryon is performed,
  !   where the possible charges were chosen according inMediumParameterset
  !   during tabulation
  !
  ! INPUTS
  ! * integer, intent(in)              :: particleID   -- ID of baryon
  ! * real, dimension(0:3), intent(in) :: Momentum     -- Momentum (in LRF)
  ! * real, intent(in)                 :: baremass     -- bare mass of baryon
  ! * real, intent(in)                 :: rhoN,rhoP    -- proton and neutron density in fm^-3
  ! * integer, intent(in)              :: switch       -- see OUTPUT for consequences of this switch
  ! * logical, optional, intent(in)    :: doMassCorrect_in
  !   --  only for debugging: switch on/off mass correction
  !
  ! OUTPUT
  ! * real :: width
  ! * logical, optional :: outOfBounds -- .true. if the input variables are out of bounds
  !
  ! If switch equals...:
  ! * 1=only collisional width
  ! * 2=only pauli blocked free width
  ! * else= sum of collisional width and pauli blocked free width
  !****************************************************************************
  function get_inMediumWidth(particleID,momentum,baremass,rhoN,rhoP,switch,doMassCorrect_in,outOfBounds) result(width)
    use idTable, only: nres,nucleon,delta
    use mediumDefinition
    use particleProperties, only: hadron
    use minkowski, only: abs4
    use vector, only: absVec
    use deltaWidth, only: delOset
    use constants, only: mN

    integer, intent(in)              :: particleID
    real, intent(in)                 :: baremass
    real, intent (in)                :: rhoN,rhoP
    logical, optional, intent(in)    :: doMassCorrect_in ! only for debugging
    logical, optional, intent(out)   :: outOfBounds
    real :: width
    real, dimension(0:3), intent(in) :: momentum
    integer, intent(in) :: switch

    integer, parameter :: showColl=1
    integer, parameter :: showPauli=2

    real  :: absP
    integer :: index_absP,index_mass,index_mass_up ,index_mass_low, index_rhoN,index_rhoP
    real :: weight
    logical,save  :: firstTime=.true.
    real, parameter :: epsilon=1E-8
    real :: mass
    !real :: p_null, pot
    !type(particle) :: part
    real :: imsig2,imsig3,imsigq
    logical :: doMassCorrect

    if (present(outOfBounds)) outOfBounds=.false.

    if (readInput_flag) call readinput

    if (particleID<minRes .or. particleID>maxres) then
      width = 0.
      return
    end if

    if (firstTime) then
       ! READ IN THE DATA FILES:
       allocate(widthTable(max(minRes,1):min(maxRes,nres+1),&
            0:numSteps_absP,0:numSteps_mass,0:numSteps_rhoN,0:numSteps_rhoP,&
            1:2))
       delta_rhoN=max_rhoN/float(numSteps_rhoN)
       delta_rhoP=max_rhoP/float(numSteps_rhoP)
       delta_absP=max_absP/float(numSteps_absP)
       call readTable
       firstTime=.false.
    end if

    doMassCorrect=.true.
    if (present(doMassCorrect_in)) doMassCorrect=doMassCorrect_in

    if (particleID.eq.nucleon) then
       ! We don't know the off-shell cross sections. Therefore we
       ! assume that the nucleon width is independent of mass:
       mass=mN
    else
       if (doMassCorrect) then
          ! Since we didn't use the potentials in the evaluation of the width,
          ! the mass must be set to the bare mass.
          mass = baremass
       else
          mass = abs4(momentum)
       end if
       if (mass.lt.hadron(particleID)%minmass) then
          width=0.
          if (present(outOfBounds)) outOfBounds=.true.
          return
       end if
    end if

    ! Absolute Momentum:
    absP=absVec(momentum(1:3))

    if (extrapolateAbsP) then
       if (absP.gt.max_absP) absP=max_absP
    end if

    if ((particleID.lt.1).or.(particleID.gt.nres+1)) then
       write(*,*) 'ERROR: particleID out of bounds in get_inMediumWidth',&
            particleID, nres+1
       call TRACEBACK()
    end if


    if (mass.lt.hadron(particleID)%minmass) then
       width=0.
       if (present(outOfBounds)) outOfBounds=.true.
       return
    end if

    if (mass.lt.min_mass(particleID)) then
       width=0.
       if (present(outOfBounds)) outOfBounds=.true.
       return
    end if

    index_absP=NINT(absP/delta_absP)
    index_mass=NINT((mass-min_mass(particleID))/delta_mass(particleID))
    index_rhoP=NINT(rhoP*0.197**3/delta_rhoP)
    index_rhoN=NINT(rhoN*0.197**3/delta_rhoN)

    if ((index_absP.le.numSteps_absP).and.(index_mass.le.numSteps_mass)&
         & .and.(index_rhoN.le.numSteps_rhoN).and.(index_rhoP.le.numSteps_rhop)&
         & .and.(index_absP.ge.0).and.(index_mass.ge.0)&
         & .and.(index_rhoN.ge.0).and.(index_rhoP.ge.0)) then

       if (particleID.eq.Delta.and.deltaOset) then
          if (switch.eq.showColl) then
             call deloset(mass,rhoN+rhoP,imsig2,imsig3,imsigq)
             width=2.*(imsig2+imsig3+imsigq) ! Collisional width
          else if (switch.eq.showPauli) then
             width=interpolate(particleID,absP,mass,rhoN,rhoP,switch) !pauli blocked free width
          else
             call deloset(mass,rhoN+rhoP,imsig2,imsig3,imsigq)
             width=2.*(imsig2+imsig3+imsigq) ! collisional width
             if (fullInterpolation) then
                width=width+interpolate(particleID,absP,mass,rhoN,rhoP,showPauli) ! +Pauli blocked free width
             else
                width=width+widthTable(particleId,index_absP,index_mass,index_rhoN,index_rhoP,showPauli)
             end if
          end if
       else
          if ((switch.eq.showColl).or.(switch.eq.showPauli)) then
             !width=widthTable(particleId,index_absP,index_mass,index_rhoN,index_rhoP,switch)
             width=interpolate(particleID,absP,mass,rhoN,rhoP,switch)
          else
             if (fullInterpolation) then
                width=interpolate(particleID,absP,mass,rhoN,rhoP)
             else
                index_mass_low=Int((mass-min_mass(particleID))/delta_mass(particleID))
                index_mass_up=index_mass_low+1
                if (index_mass_up.le.numSteps_mass.and.index_mass_low.ge.0.and.mass.gt.min_mass(particleID)) then
                   ! linear interpolation in mass
                   weight=1.-(mass-min_mass(particleID)-float(index_mass_low)*delta_mass(particleID))/delta_mass(particleID)
                   if (weight.gt.1+epsilon.or.weight.lt.0.-epsilon) then
                      write(*,*) 'weight error',weight,index_mass_low,&
                           &index_mass_up,mass,min_mass(particleID),delta_mass(particleID)
                      call TRACEBACK()
                   end if
                   width=weight*sum(widthTable(particleId,index_absP,index_mass_low,index_rhoN,index_rhoP,:))&
                        & + (1.-weight)*sum(widthTable(particleId,index_absP,index_mass_up,index_rhoN,index_rhoP,:))
                else
                   width=sum(widthTable(particleId,index_absP,index_mass,index_rhoN,index_rhoP,:))
                end if
             end if
          end if
       end if
    else
       if (mass.lt.hadron(particleID)%minmass) then
          width=0.
          return
       else if ((index_mass.gt.numSteps_mass).and.(mass.lt.max_mass(particleID)+0.2)) then
          width=0.
          if (debugFlag) write(*,*) 'WARNING: Out of bounds in get_inMediumWidth'
          if (debugFlag) write(*,*) "mass too large", mass, " > ", max_mass(particleID)
       else
          write(*,*) 'WARNING: Out of bounds in get_inMediumWidth',particleID
          if (index_absP.gt.numSteps_absP) &
               write(*,*) "momentum too large", absP, " > ", max_absP
          if (index_rhoN.gt.numSteps_rhoN) &
               write(*,*) "rhoN too large", &
               & rhoN, " > ", max_rhoN/0.197**3," fm^-3"
          if (index_rhoP.gt.numSteps_rhop) &
               write(*,*) "rhoP too large", &
               & rhoP, " > ", max_rhoP/0.197**3," fm^-3"
          if (index_mass.gt.numSteps_mass) &
               write(*,*) "mass too large", mass, " > ", max_mass(particleID)
          if (index_absP.lt.0) &
               write(*,*) "absP too small", absP
          if (index_mass.lt.0) &
               write(*,*) "mass too small", mass, " < ", min_mass(particleID)
          if (index_rhoN.lt.0) &
               write(*,*) "rhoN too small", rhoN*0.197**3
          if (index_rhoP.lt.0) &
               write(*,*) "rhoP too small", rhoP*0.197**3
          width=0.
       end if
       if (present(outOfBounds)) outOfBounds=.true.
    end if

    width=max(minimalWidth,width)

  contains

    real function interpolate(particleID,absP,mass,rhoN,rhoP,channel)
      use tabulation, only: interpolate4

      integer,intent(in) :: particleID
      real,intent(in) :: absP,mass,rhoN,rhoP
      integer,intent(in), optional :: channel
      ! If present, then only the result of this channel will
      ! be interpolated, otherwise the sum of both

      real(4), dimension(1:4) :: x
      !real, save, dimension(0:numSteps_rhoN)  :: A3
      !real, save, dimension(0:numSteps_rhoP)  :: A4
      !real, save, dimension(0:numSteps_absP)  :: A1
      !real, save, dimension(0:numSteps_mass)  :: A2_template
      !real, dimension(0:numSteps_mass)  :: A2

      logical,save :: first=.true.

      !integer :: i,j
      !integer :: rhoP_index,rhoN_index

      real(4), dimension(1:4),save :: min, max, delta
      x(1:4)=(/absP,mass,rhoN*0.197**3,rhoP*0.197**3/)


      ! Use an interpolation method which assumes constant grid distances
      if (first) then
         ! Initialize arrays containing grid information
         min  =(/ 0., min_mass(particleID), 0., 0./)
         max  =(/ float(numSteps_absP)*delta_absP, &
              &   float(numSteps_mass)*delta_Mass(particleID)+min_mass(particleID), &
              &   float(numSteps_rhoN)*delta_rhoN, &
              &   float(numSteps_rhoP)*delta_rhoP /)
         delta=(/ delta_absP, delta_Mass(particleID), delta_rhoN, delta_rhoP /)
         first=.false.
      else
         ! Only update mass parameters of the grid information
         min(2)  =min_mass(particleID)
         max(2)  =float(numSteps_mass) * delta_Mass(particleID)+min_mass(particleID)
         delta(2)=delta_Mass(particleID)
      end if
      if (present(channel)) then
         interpolate=interpolate4(X,min,max,delta,widthTable(particleId,:,:,:,:,channel))
      else
         interpolate=interpolate4(X,min,max,delta,widthTable(particleId,:,:,:,:,1)) &
              &     +interpolate4(X,min,max,delta,widthTable(particleId,:,:,:,:,2))
      end if

    end function interpolate

    !**************************************************************************
    !****s* get_inMediumWidth/readTable
    ! NAME
    ! subroutine readTable
    !
    ! PURPOSE
    ! Reads tabulated arrays from files:
    ! *  buuinput/inMediumWidth/InMediumWidth."particleID"_1_?.dat.bz2 and
    ! *  buuinput/inMediumWidth/InMediumWidth."particleID"_2_?.dat.bz2 .
    ! OUTPUT
    ! Initialized arrays widthTable, min_mass, max_mass, delta_mass
    !
    ! NOTES
    ! The file name may be given via the jobcard, see fileNameMask
    !**************************************************************************
    subroutine readTable
      use output, only: intToChar,intToChar1,Write_ReadingInput
      use bzip

      character(5) :: raute
      character(1000) :: fileName

      integer :: particleID,index_absP,index_mass,index_rhoN!,index_rhoP
      integer :: ios, i
      logical :: fileFailure
      integer :: numSteps_absP_in, numSteps_mass_in, &
           numSteps_rhoN_in, numSteps_rhoP_in, &
           num_MonteCarlo_Points_in,min_charge_loop_in,max_charge_loop_in
      real :: max_absP_in, max_rhoP_in , max_rhoN_in, minMass_in, maxMass_in
      type(bzFile) :: f
      character(len=100) :: buf
      integer :: ll


      ! Set the bounds for the parametrization of the mass
      do particleId=1,nres+1
         !         min_mass(particleID)=max(minimalMass(particleID)+0.02,baryon(particleID)%mass-max(0.1,baryon(particleID)%width*3.))
         !         max_mass(particleID)=baryon(particleID)%mass+max(0.1,baryon(particleID)%width*3.)

         min_mass(particleID) = get_minMass(particleID)
         max_mass(particleID) = 7. ! TODO: should be 4.
         delta_mass(particleID)=(max_mass(particleID)-min_mass(particleID))/float(numSteps_mass)
      end do


      call Write_ReadingInput("InMediumWidth (baryon)",0)

      fileFailure=.false.
      outerLoop: do i=1,2
         idLoop: do particleId=max(minRes,1),min(maxRes,nres+1)
            if (onlyNucleon) then
               if (particleID.ge.2) cycle
            end if
            fileName=trim(FileNameMask)//"." &
                 //trim(intToChar(particleID)) &
                 //'_'//trim(intToChar1(i)) &
                 //'_'//trim(intToChar1(inMediumParameterset)) &
                 //'.dat.bz2'

            f = bzOpenR(trim(fileName),iostat=ios)
            if (ios.ne.0) then
               write(*,*) 'Error in opening input file: ',trim(fileName)
               fileFailure=.true.
               exit outerLoop
            end if
            ll = 0
            call bzReadLine(f,buf,ll)
            read(buf(1:ll),*,iostat=ios) raute, &
                 numSteps_absP_in, numSteps_mass_in, &
                 numSteps_rhoN_in,numSteps_rhoP_in,num_MonteCarlo_Points_in, &
                 min_charge_loop_in,max_charge_loop_in
            if (ios.ne.0) then
               write(*,*) 'Error in reading input file (2): ',trim(fileName)
               fileFailure=.true.
               exit outerLoop
            end if
            ll = 0
            call bzReadLine(f,buf,ll)
            read(buf(1:ll),*,iostat=ios) raute, max_absP_in, max_rhoP_in, &
                 max_rhoN_in, minMass_in, maxMass_in
            if (ios.ne.0) then
               write(*,*) 'Error in reading input file (3): ',trim(fileName)
               fileFailure=.true.
               exit outerLoop
            end if
            write(*,'(A,I3,A,F8.6,A,F8.6,A,F8.6,A,F8.6,A,I2,A,I2,A,I4)') &
                 'ID=',particleID, &
                 '   minM=',minMass_in,' maxM=',maxMass_in,' dM=',delta_mass(particleID), &
                 '   maxP=',max_absP_in, &
                 '   minCh=',min_charge_loop_in,' maxCh=',max_charge_loop_in, &
                 '   MC_points=',num_MonteCarlo_Points_in
            if ((numSteps_absP.ne.numSteps_absP_in) &
                 & .or.(numSteps_rhoP.ne.numSteps_rhoP_in) &
                 & .or.(numSteps_rhoN.ne.numSteps_rhoN_in) &
                 & .or.(numSteps_mass.ne.numSteps_mass_in) &
                 & .or.(min_charge_loop_in.ne.get_min_charge_loop()) &
                 & .or.(max_charge_loop_in.ne.get_max_charge_loop()) &
                 & .or.(abs(max_absP_in-max_absP).gt.0.00001) &
                 & .or.(abs(max_rhoP_in-max_rhoP).gt.0.00001) &
                 & .or.(abs(max_rhoN_in-max_rhoN).gt.0.00001) &
                 & .or.(abs(minMass_in-min_mass(particleID)).gt.0.00001) &
                 & .or.(abs(maxMass_in-max_mass(particleID)).gt.0.00001) &
                 & ) then
               write(*,*) 'Error in reading input file (4): ',trim(fileName)
               write(*,*) 'Grid or number of Monte Carlo points differ!'
               if (numSteps_absP.ne.numSteps_absP_in) &
                    write(*,*) 'absP',numSteps_absP,numSteps_absP_in
               if (numSteps_rhoP.ne.numSteps_rhoP_in) &
                    write(*,*) 'rhoP',numSteps_rhoP,numSteps_rhoP_in
               if (numSteps_rhoN.ne.numSteps_rhoN_in) &
                    write(*,*) 'rhoN',numSteps_rhoN,numSteps_rhoN_in
               if (numSteps_mass.ne.numSteps_mass_in) &
                    write(*,*) 'mass',numSteps_mass,numSteps_mass_in
               if (min_charge_loop_in.ne.get_min_charge_loop()) &
                    write(*,*) 'charge loop min',min_charge_loop_in,get_min_charge_loop()
               if (max_charge_loop_in.ne.get_max_charge_loop()) &
                    write(*,*) 'charge loop max',max_charge_loop_in,get_max_charge_loop()
               if (abs(max_absP_in-max_absP).gt.0.00001) &
                    write(*,*) 'max_absp', max_absP,max_absP_in
               if (abs(max_rhoP_in-max_rhoP).gt.0.00001) &
                    write(*,*) 'max_rhop', max_rhoP,max_rhoP_in
               if (abs(max_rhoN_in-max_rhoN).gt.0.00001) &
                    write(*,*) 'max_rhon', max_rhoN,max_rhoN_in
               if (abs(minMass_in-min_mass(particleID)).gt.0.00001) &
                    write(*,*) 'min_mass', minMass_in,min_mass(particleID)
               if (abs(maxMass_in-max_mass(particleID)).gt.0.00001) &
                    write(*,*) 'max_mass', maxMass_in,max_mass(particleID)
               fileFailure=.true.
               exit outerLoop
            end if

            ll = 0
            do index_absP=0,numSteps_absP
               do index_mass=0,numSteps_mass
                  do index_rhoN=0,numSteps_rhoN
                     call bzReadLine(f,buf,ll)
                     read(buf(1:ll),*,iostat=ios) widthTable(particleID,index_absP,index_mass,index_rhoN,0:numSteps_rhoP,i)
                     if (IOS.ne.0) then
                        write(*,*) 'Error in reading input file: ',fileName
                        write(*,*) 'Error', particleID,ios,index_absP,index_mass,index_rhoN
                        fileFailure=.true.
                        exit outerLoop
                     end if
                  end do
               end do
            end do
         end do idLoop
      end do outerLoop
      call bzCloseR(f)
      if (fileFailure) then
         write(*,*) 'You first need to tabulate the IN-MEDIUM WIDTH table. STOP!!'
         call TRACEBACK()
      end if

      call Write_ReadingInput("InMediumWidth (baryon)",1)

    end subroutine readTable


  end function get_inMediumWidth

  !****************************************************************************
  !****s* baryonWidthMedium_tables/dumpTable
  ! NAME
  ! subroutine dumpTable(iFile,particleID,iAbsP,iMass,iRhoN,iRhoP, iColumn)
  !
  ! PURPOSE
  ! write table in special format for plotting
  !
  ! INPUTS
  ! * integer :: iFile  -- output file
  ! * integer :: particleID -- ID of baryon
  ! * integer :: iAbsP -- bin of absP to use
  ! * integer :: iMass -- bin of mass to use
  ! * integer :: iRhoN -- bin of rhoN to use
  ! * integer :: iRhoP -- bin of rhoP to use
  ! * integer :: iColumn -- select the output, see below
  !
  ! OUTPUT
  ! according the value of iColumn, the running variable is
  ! * 1: absP
  ! * 2: mass
  ! * 12,21: 2D-Array, absP, mass
  !
  ! NOTES
  ! * if iColumn=1, then the value iMass=-1 indicates, that the average over
  !   all masses is printed (this is useful for the nucleon only, where the
  !   width does not depend on the mass)
  !****************************************************************************
  subroutine dumpTable(iFile,particleID,iAbsP,iMass,iRhoN,iRhoP, iColumn)
    integer, intent(in) :: iFile,particleID,iAbsP,iMass,iRhoN,iRhoP, iColumn

    integer :: iVal,iVal2
    real :: absP,Mass,RhoN,RhoP, dummy, v1,v2

    ! initialization:
    dummy = get_inMediumWidth(particleID, (/0.,0.,0.,0./), 0.,0.,0.,3)

    select case(iColumn)
    case (1)
       if ((iMass<-1).or.(iMass>numSteps_mass)) then
          write(*,*) "iMass = ",iMass
          call TRACEBACK("wrong iMass")
       end if
       if ((iRhoN<0).or.(iRhoP<0) &
            .or.(iRhoN>numSteps_rhoN).or.(iRhoP>numSteps_rhoP)) then
          write(*,*) "iRhoN, iRhoP = ",iRhoN,iRhoP
          call TRACEBACK("wrong iRho")
       end if

       ! absP=float(iAbsP)*delta_absP
       if (iMass == -1) then
          mass = 0.0
       else
          mass=min_mass(particleID)+float(iMass)*delta_mass(particleID)
       end if
       rhoN=float(iRhoN)*delta_rhoN
       rhoP=float(iRhoP)*delta_rhoP

       do iVal = 0,numSteps_absP
          absP=float(iVal)*delta_absP

          if (iMass == -1) then
             v1 = sum(widthTable(particleID,iVal,:,iRhoN,iRhoP,1))/(numSteps_mass+1)
             v2 = sum(widthTable(particleID,iVal,:,iRhoN,iRhoP,2))/(numSteps_mass+1)
          else
             v1 = widthTable(particleID,iVal,iMass,iRhoN,iRhoP,1)
             v2 = widthTable(particleID,iVal,iMass,iRhoN,iRhoP,2)
          end if
          write(iFile,*) absP,mass,rhoN,rhoP,v1,v2

       end do

    case (2)

       if ((iAbsP<0).or.(iAbsP>numSteps_absP)) then
          write(*,*) "iAbsP = ",iAbsP
          call TRACEBACK("wrong iAbsP")
       end if
       if ((iRhoN<0).or.(iRhoP<0) &
            .or.(iRhoN>numSteps_rhoN).or.(iRhoP>numSteps_rhoP)) then
          write(*,*) "iRhoN, iRhoP = ",iRhoN,iRhoP
          call TRACEBACK("wrong iRho")
       end if

       absP=float(iAbsP)*delta_absP
       ! mass=min_mass(particleID)+float(iMass)*delta_mass(particleID)
       rhoN=float(iRhoN)*delta_rhoN
       rhoP=float(iRhoP)*delta_rhoP

       do iVal = 0,numSteps_mass
          mass=min_mass(particleID)+float(iVal)*delta_mass(particleID)

          write(iFile,*) absP,mass,rhoN,rhoP,&
               widthTable(particleID,iAbsP,iVal,iRhoN,iRhoP,1), &
               widthTable(particleID,iAbsP,iVal,iRhoN,iRhoP,2)
       end do

    case (12,21)

       if ((iRhoN<0).or.(iRhoP<0) &
            .or.(iRhoN>numSteps_rhoN).or.(iRhoP>numSteps_rhoP)) then
          write(*,*) "iRhoN, iRhoP = ",iRhoN,iRhoP
          call TRACEBACK("wrong iRho")
       end if

       ! absP=float(iAbsP)*delta_absP
       ! mass=min_mass(particleID)+float(iMass)*delta_mass(particleID)
       rhoN=float(iRhoN)*delta_rhoN
       rhoP=float(iRhoP)*delta_rhoP

       do iVal = 0,numSteps_mass
          mass=min_mass(particleID)+float(iVal)*delta_mass(particleID)

          do iVal2 = 0,numSteps_absP
             absP=float(iVal2)*delta_absP

             write(iFile,*) absP,mass,rhoN,rhoP,&
                  widthTable(particleID,iVal2,iVal,iRhoN,iRhoP,1), &
                  widthTable(particleID,iVal2,iVal,iRhoN,iRhoP,2)
          end do
          write(iFile,*)
       end do

    case default
       call TRACEBACK('wrong column selection')

    end select
    write(iFile,*)

  end subroutine dumpTable



end module baryonWidthMedium_tables
