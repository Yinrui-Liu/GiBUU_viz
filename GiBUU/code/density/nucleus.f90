!******************************************************************************
!****m* /nucleus
! NAME
! module nucleus
!
! PURPOSE
! Provides access to the projectile and target nuclei.
!
! INPUTS
! Namelists "target" and "projectile".
!******************************************************************************
module nucleus

  use nucleusDefinition, only: tNucleus
  use CallStack, only: TRACEBACK


  implicit none
  private

  public :: getProjectile
  public :: getTarget

  type(tNucleus), save, pointer :: targetNuc     => NULL()  ! target nucleus
  type(tNucleus), save, pointer :: projectileNuc => NULL()  ! projectile nucleus

  integer, parameter :: Zmax = 111  ! maximum Z for defaultIsotope

  ! The following array contains a default mass number A for each nuclear
  ! charge number Z (given as index), corresponding to a "default isotope".
  ! The default isotope is chosen by the folling scheme:
  ! * for stable elements: either naturally most abundant isotope or weighted
  !   average according to natural abundances (rounded to nearest integer)
  ! * for unstable elements: most stable isotope
  integer, dimension(0:Zmax), parameter :: defaultIsotope = (/   1, & !   0 = n
                                                                 1, & !   1 = H
                                                                 4, & !   2 = He
                                                                 7, & !   3 = Li
                                                                 9, & !   4 = Be
                                                                11, & !   5 = B
                                                                12, & !   6 = C
                                                                14, & !   7 = N
                                                                16, & !   8 = O
                                                                19, & !   9 = F
                                                                20, & !  10 = Ne
                                                                23, & !  11 = Na
                                                                24, & !  12 = Mg
                                                                27, & !  13 = Al
                                                                28, & !  14 = Si
                                                                31, & !  15 = P
                                                                32, & !  16 = S
                                                                35, & !  17 = Cl
                                                                40, & !  18 = Ar
                                                                39, & !  19 = K
                                                                40, & !  20 = Ca
                                                                45, & !  21 = Sc
                                                                48, & !  22 = Ti
                                                                51, & !  23 = V
                                                                52, & !  24 = Cr
                                                                55, & !  25 = Mn
                                                                56, & !  26 = Fe
                                                                59, & !  27 = Co
                                                                59, & !  28 = Ni
                                                                63, & !  29 = Cu
                                                                65, & !  30 = Zn
                                                                70, & !  31 = Ga
                                                                73, & !  32 = Ge
                                                                75, & !  33 = As
                                                                79, & !  34 = Se
                                                                80, & !  35 = Br
                                                                84, & !  36 = Kr
                                                                85, & !  37 = Rb
                                                                88, & !  38 = Sr
                                                                89, & !  39 = Y
                                                                91, & !  40 = Zr
                                                                93, & !  41 = Nb
                                                                96, & !  42 = Mo
                                                                98, & !  43 = Tc
                                                               101, & !  44 = Ru
                                                               103, & !  45 = Rh
                                                               106, & !  46 = Pd
                                                               109, & !  47 = Ag
                                                               112, & !  48 = Cd
                                                               115, & !  49 = In
                                                               120, & !  50 = Sn
                                                               122, & !  51 = Sb
                                                               128, & !  52 = Te
                                                               127, & !  53 = I
                                                               131, & !  54 = Xe
                                                               133, & !  55 = Cs
                                                               137, & !  56 = Ba
                                                               139, & !  57 = La
                                                               142, & !  58 = Ce
                                                               141, & !  59 = Pr
                                                               144, & !  60 = Nd
                                                               145, & !  61 = Pm
                                                               150, & !  62 = Sm
                                                               152, & !  63 = Eu
                                                               157, & !  64 = Gd
                                                               159, & !  65 = Tb
                                                               163, & !  66 = Dy
                                                               165, & !  67 = Ho
                                                               167, & !  68 = Er
                                                               169, & !  69 = Tm
                                                               173, & !  70 = Yb
                                                               175, & !  71 = Lu
                                                               178, & !  72 = Hf
                                                               181, & !  73 = Ta
                                                               184, & !  74 = W
                                                               186, & !  75 = Re
                                                               190, & !  76 = Os
                                                               192, & !  77 = Ir
                                                               195, & !  78 = Pt
                                                               197, & !  79 = Au
                                                               201, & !  80 = Hg
                                                               204, & !  81 = Tl
                                                               208, & !  82 = Pb
                                                               209, & !  83 = Bi
                                                               209, & !  84 = Po
                                                               210, & !  85 = At
                                                               222, & !  86 = Rn
                                                               223, & !  87 = Fr
                                                               226, & !  88 = Ra
                                                               227, & !  89 = Ac
                                                               232, & !  90 = Th
                                                               231, & !  91 = Pa
                                                               238, & !  92 = U
                                                               237, & !  93 = Np
                                                               244, & !  94 = Pu
                                                               243, & !  95 = Am
                                                               247, & !  96 = Cm
                                                               247, & !  97 = Bk
                                                               251, & !  98 = Cf
                                                               252, & !  99 = Es
                                                               257, & ! 100 = Fm
                                                               258, & ! 101 = Md
                                                               259, & ! 102 = No
                                                               262, & ! 103 = Lr
                                                               267, & ! 104 = Rf
                                                               268, & ! 105 = Db
                                                               271, & ! 106 = Sg
                                                               270, & ! 107 = Bh
                                                               269, & ! 108 = Hs
                                                               276, & ! 109 = Mt
                                                               281, & ! 110 = Ds
                                                               280 /) ! 111 = Rg


contains

  !****************************************************************************
  !****f* nucleus/getTarget
  ! NAME
  ! function getTarget()
  ! PURPOSE
  ! Returns an initialized target nucleus resting at 0. with velocity=0.
  ! INPUTS
  ! * NONE
  ! OUTPUT
  ! * type(nucleus) :: getTarget
  !****************************************************************************
  function getTarget()
    use nucleusDefinition
    type(tnucleus),pointer :: getTarget
    if (.not. associated(targetNuc)) then
      allocate(targetNuc)
      call initTarget
    end if
    getTarget => targetNuc
  end function getTarget


  !****************************************************************************
  !****f* nucleus/getProjectile
  ! NAME
  ! function getProjectile()
  ! PURPOSE
  ! Returns an initialized projectile nucleus resting at 0. with velocity=0.
  ! INPUTS
  ! * NONE
  ! OUTPUT
  ! * type(nucleus) :: getProjectile
  !****************************************************************************
  function getProjectile()
    use nucleusDefinition
    type(tnucleus),pointer :: getProjectile
    if (.not. associated(projectileNuc)) then
      allocate(ProjectileNuc)
      call initProjectile
    end if
    getProjectile => ProjectileNuc
  end function getProjectile


  !****************************************************************************
  !****s* nucleus/initProjectile
  ! NAME
  ! subroutine initProjectile
  ! PURPOSE
  ! Initializes the projectile nucleus resting in the origin
  ! according to Information in namelist "projectile" in jobcard.
  !****************************************************************************
  subroutine initProjectile
    use output, only: Write_ReadingInput

    !**************************************************************************
    !****g* initProjectile/A
    ! SOURCE
    integer, save :: A = 0
    ! PURPOSE
    ! Mass A of projectile nucleus ( = number of nucleons).
    ! If zero, a default isotope is chosen for the given Z.
    !**************************************************************************

    !**************************************************************************
    !****g* initProjectile/Z
    ! SOURCE
    integer, save :: Z = 20
    ! PURPOSE
    ! Charge Z of projectile nucleus ( = number of protons).
    !**************************************************************************

    !**************************************************************************
    !****g* initProjectile/Projectile_A
    ! SOURCE
    integer, save :: Projectile_A = -99
    ! PURPOSE
    ! deprecated, use 'A' instead
    !**************************************************************************

    !**************************************************************************
    !****g* initProjectile/Projectile_Z
    ! SOURCE
    integer, save :: Projectile_Z = -99
    ! PURPOSE
    ! deprecated, use 'Z' instead
    !**************************************************************************

    !**************************************************************************
    !****g* initProjectile/fermiMotion
    ! SOURCE
    logical,save :: fermiMotion=.true.
    ! PURPOSE
    ! Determines whether particles in projectile nucleus feel Fermi motion.
    !**************************************************************************

    !**************************************************************************
    !****g* initProjectile/densitySwitch_static
    ! SOURCE
    !
    integer,save :: densitySwitch_static=3
    ! PURPOSE
    ! This switch is important, because it decides, which static density is
    ! used to set up the testparticles in the nuclei before the first
    ! time-step.
    !
    ! Possible values:
    ! * 0 : density=0.0
    ! * 1 : Static density using Woods-Saxon according to H. Lenske
    ! * 2 : Static density according to Nieves93
    ! * 3 : Static density using Woods-Saxon according to H. Lenske,
    !   different radii for neutrons and protons
    ! * 4 : Static density according oscillator shell model
    ! * 5 : Density distribution is a sphere with density according to the
    !   input value of "fermiMomentum_input".
    ! * 6 : Static Density based on LDA, implemented by Birger Steinmueller
    ! * 7 : Static Density based on LDA + Welke potential
    ! * 8 : Static Density prescription according Relativistic Thomas-Fermi
    !   (valid only in RMF-mode)
    !
    ! Possible nuclei for the different prescriptions:
    ! * 1 : A > 2 (only A > 16 makes sense)
    ! * 2 : Be(9), C(12), O(16,18), Al(27), Ar(40), Ca(40,44), Fe(56), Cu(63),
    !   As(75), Ce(142), Sn(112,116,120,124), Ta(181), Au(197), Pb(208)
    !   see densityStatic.f90, subroutine denspar for more info
    ! * 3 : 6->C(12), 8->O(16),O(18), 13->Al(27), 20->Ca(40),Ca(44), 79->Au(197)
    !   82->Pb(208)
    ! * 4: 2->He(4), 4->Be(9), 5->B(11), 6->C(12), 8->O(16)
    !
    ! Refs:
    ! * J.Nieves, E.Oset, C.Garcia-Recio, Nucl.Phys.A 554 (1993) 509
    !**************************************************************************

    !**************************************************************************
    !****g* initProjectile/fermiMomentum_input
    ! SOURCE
    !
    real,save :: fermiMomentum_input=0.225
    ! PURPOSE
    ! Input value of the fermi momentum for densitySwitch_static=5 (in GeV).
    !**************************************************************************

    !**************************************************************************
    !****g* initProjectile/anti
    ! SOURCE
    !
    logical,save :: anti=.FALSE.
    ! PURPOSE
    ! Indicate, whether it is a anti-nucleus
    !**************************************************************************

    !**************************************************************************
    !****n* initProjectile/projectile
    ! NAME
    ! NAMELIST /projectile/
    !
    ! PURPOSE
    ! Includes the input parameters for the projectile nucleus:
    ! * A
    ! * Z
    ! * Projectile_A (deprecated)
    ! * Projectile_Z (deprecated)
    ! * fermiMotion
    ! * densitySwitch_static
    ! * fermiMomentum_input
    ! * anti
    !**************************************************************************
    NAMELIST /projectile/ A, Z, Projectile_A, Projectile_Z, fermiMotion, &
         densitySwitch_static, fermiMomentum_input, anti

    integer :: ios

    if (.not.projectileNuc%DoInit) return

    call Write_ReadingInput('projectile',0)
    rewind(5)
    read(5,nml=projectile,iostat=ios)

    if (projectile_A > 0) then
       write(*,*) "parameter 'projectile_A' is deprecated. Please start using simply 'A' instead."
       A = projectile_A
    end if
    if (projectile_Z >= 0) then
       write(*,*) "parameter 'projectile_Z' is deprecated. Please start using simply 'Z' instead."
       Z = projectile_Z
    end if

    if (A<=0 .and. Z>=0 .and. Z<=Zmax) A = defaultIsotope(Z)

    call Write_ReadingInput('projectile',0,ios)
    write(*,*) 'A=',A,' Z=',Z
    write(*,*) 'fermiMotion=', fermiMotion
    write(*,*) 'densitySwitch_static = ',densitySwitch_static
    if (densitySwitch_static==5) then
       write(*,*) ' -> A naive Fermi gas distribution is initialized. No surface - just a sphere!'
       write(*,*) '    Fermi momentum = ',fermiMomentum_input
       projectileNuc%fermiMomentum_input=fermiMomentum_input
    end if

    if (anti) then
       write(*,*) 'The nucleus is an anti-nucleus!'
       if (A>2) call TRACEBACK("anti works only for nucleon and deuterium!")
    end if

    projectileNuc%mass   = A
    projectileNuc%charge = Z
    projectileNuc%fermiMotion=fermiMotion
    projectileNuc%densitySwitch_static=densitySwitch_static
    projectileNuc%anti = anti

    call InitNucleus(projectileNuc,fermiMomentum_input)

    projectileNuc%DoInit = .false.

    !**************************************************************************
    !****o* nucleus/DensTab_projectile.dat
    ! NAME
    ! file DensTab_projectile.dat
    ! PURPOSE
    ! Density tabulation of the projectile nucleus at initialization.
    !**************************************************************************
    call projectileNuc%WriteStaticDens('DensTab_projectile.dat')

    call Write_ReadingInput('projectile',2)

  end subroutine initProjectile


  !****************************************************************************
  !****s* nucleus/initTarget
  ! NAME
  ! subroutine initTarget
  ! PURPOSE
  ! Initializes the target nucleus resting in the origin
  ! according to Information in namelist "target" in jobcard.
  !****************************************************************************
  subroutine initTarget
    use output, only: Write_ReadingInput
    use RMF, only: getRMF_flag
    use inputGeneral, only: eventType
    use eventtypes, only: HeavyIon

    !**************************************************************************
    !****g* initTarget/A
    ! SOURCE
    integer, save :: A = 0
    ! PURPOSE
    ! Mass A of target nucleus ( = number of nucleons).
    ! If zero, a default isotope is chosen for the given Z.
    !**************************************************************************

    !**************************************************************************
    !****g* initTarget/Z
    ! SOURCE
    integer, save :: Z = 20
    ! PURPOSE
    ! Charge Z of target nucleus ( = number of protons).
    !**************************************************************************

    !**************************************************************************
    !****g* initTarget/Target_A
    ! SOURCE
    integer, save :: Target_A = -99
    ! PURPOSE
    ! deprecated, use 'A' instead
    !**************************************************************************

    !**************************************************************************
    !****g* initTarget/Target_Z
    ! SOURCE
    integer, save :: Target_Z = -99
    ! PURPOSE
    ! deprecated, use 'Z' instead
    !**************************************************************************

    !**************************************************************************
    !****g* initTarget/fermiMotion
    ! SOURCE
    logical,save :: fermiMotion=.true.
    ! PURPOSE
    ! Determines whether particles in target nucleus feel Fermi motion.
    !**************************************************************************

    !**************************************************************************
    !****g* initTarget/densitySwitch_static
    ! SOURCE
    !
    integer,save :: densitySwitch_static=3
    ! PURPOSE
    ! This switch is important, because it decides, which static density is
    ! used to set up the testparticles in the nuclei before the first
    ! time-step.
    !
    ! Possible values:
    ! * 0 : density=0.0
    ! * 1 : Static density using Woods-Saxon according to H. Lenske
    ! * 2 : Static density according to Nieves93
    ! * 3 : Static density using Woods-Saxon according to H. Lenske,
    !   different radii for neutrons and protons
    ! * 4 : Static density according oscillator shell model
    ! * 5 : Density distribution is a sphere with density according to the
    !   input value of "fermiMomentum_input".
    ! * 6 : Static density based on LDA, implemented by Birger Steinmueller
    ! * 7 : Static density based on LDA + Welke potential
    ! * 8 : Static density prescription according Relativistic Thomas-Fermi
    !   (valid only in RMF-mode)
    !
    ! Possible nuclei for the different prescriptions:
    ! * 1 : A > 2 (only A > 16 makes sense)
    ! * 2 : Be(9), C(12), O(16,18), Al(27), Ar(40), Ca(40,44), Fe(56), Cu(63),
    !   As(75), Ce(142), Sn(112,116,120,124), Ta(181), Au(197), Pb(208)
    !   see densityStatic.f90, subroutine denspar for more info
    ! * 3 : 6->C(12), 8->O(16),O(18), 13->Al(27), 20->Ca(40),Ca(44), 79->Au(197)
    !   82->Pb(208)
    ! * 4: 2->He(4), 4->Be(9), 5->B(11), 6->C(12), 8->O(16)
    !
    ! Refs:
    ! * J.Nieves, E.Oset, C.Garcia-Recio, Nucl.Phys.A 554 (1993) 509
    !**************************************************************************

    !**************************************************************************
    !****g* initTarget/fermiMomentum_input
    ! SOURCE
    !
    real,save :: fermiMomentum_input=0.225
    ! PURPOSE
    ! Input value of the fermi momentum for densitySwitch_static=5 (in GeV).
    !**************************************************************************

    !**************************************************************************
    !****g* initTarget/radius_input
    ! SOURCE
    !
    real,dimension(1:2),save :: radius_input = -99.9
    ! PURPOSE
    ! Input value(s) of the radius of the nucleus in a Woods-Saxon
    ! parametrization.
    !
    ! If densitySwitch_static=...:
    ! * 1: one value is used both for protons and neutrons.
    ! * 3: two values: 1) proton, 2) neutron
    !
    ! Overrides default values only if given and positive.
    !**************************************************************************

    !**************************************************************************
    !****g* initTarget/surface_input
    ! SOURCE
    !
    real,dimension(1:2),save :: surface_input = -99.9
    ! PURPOSE
    ! Input value(s) of the surface of the nucleus in a Woods-Saxon
    ! parametrization
    !
    ! If densitySwitch_static=...:
    ! * 1: one value is used both for protons and neutrons.
    ! * 3: two values: 1) proton, 2) neutron
    !
    ! Overrides default values only if given and positive.
    !**************************************************************************

    !**************************************************************************
    !****g* initTarget/ReAdjustForConstBinding
    ! SOURCE
    !
    logical, save :: ReAdjustForConstBinding = .true.
    ! PURPOSE
    ! If this flag is set to true, we use the selected density distribution
    ! only for a preliminary step, where we calculate the baryonic potential
    ! as function of r (which depends on the density distribution).
    ! From the condition, that the binding energy has to be constant, we
    ! deduce the distribution of the fermi momentum and thus the 'new'
    ! density distribution.
    !
    ! The tabulated density distribution is replaced via the 'new' one
    ! and all behaviour is as usual.
    !
    ! Only valid in Skyrme mode (and also not for HeavyIon init).
    !**************************************************************************

    !**************************************************************************
    !****g* initTarget/ConstBinding
    ! SOURCE
    !
    real, save :: ConstBinding = -0.008
    ! PURPOSE
    ! if 'ReAdjustForConstBinding' equals true, we a trying to readjust
    ! the fermi momentum and the density such, we quarantee this value
    ! for the binding energy (in GeV).
    !
    ! Only valid in Skyrme mode (and also not for HeavyIon init)
    !**************************************************************************

    !**************************************************************************
    !****g* initTarget/doPrintGlauber
    ! SOURCE
    !
    logical, save :: doPrintGlauber = .false.
    ! PURPOSE
    ! switch on/off printing of simple Glauber calculations after init
    ! returning sigma_hA as function of sigma_hN
    ! (cf. Falter PhD, eq. (5.15))
    !**************************************************************************

    !**************************************************************************
    !****n* initTarget/target
    ! NAME
    ! NAMELIST /target/
    !
    ! PURPOSE
    ! Includes the input parameters for the target nucleus:
    ! * A
    ! * Z
    ! * Target_A (deprecated)
    ! * Target_Z (deprectaed)
    ! * fermiMotion
    ! * densitySwitch_static
    ! * fermiMomentum_input
    ! * ReAdjustForConstBinding
    ! * ConstBinding
    ! * radius_input
    ! * surface_input
    ! * doPrintGlauber
    !**************************************************************************
    NAMELIST /target/ A, Z, Target_A, Target_Z, fermiMotion, &
         densitySwitch_static, fermiMomentum_input, &
         ReAdjustForConstBinding, ConstBinding, &
         radius_input, surface_input, doPrintGlauber

    integer :: ios

    if (.not.targetNuc%DoInit) return

    call Write_ReadingInput('target',0)
    rewind(5)
    read(5,nml=target,iostat=ios)
    call Write_ReadingInput('target',0,ios)

    if (target_A > 0) then
       write(*,*) "parameter 'target_A' is deprecated. Please start using simply 'A' instead."
       A = target_A
    end if
    if (target_Z >= 0) then
       write(*,*) "parameter 'target_Z' is deprecated. Please start using simply 'Z' instead."
       Z = target_Z
    end if


    if (A<=0 .and. Z>=0 .and. Z<=Zmax) A = defaultIsotope(Z)

    write(*,*) 'A=',A,' Z=',Z
    write(*,*) 'fermiMotion = ', fermiMotion
    write(*,*) 'densitySwitch_static = ',densitySwitch_static
    if (densitySwitch_static==5) then
       write(*,*) ' -> A naive Fermi gas distribution is initialized. No surface - just a sphere!'
       write(*,*) '    Fermi momentum = ',fermiMomentum_input
       targetNuc%fermiMomentum_input=fermiMomentum_input
    end if

    targetNuc%mass   = A
    targetNuc%charge = Z
    targetNuc%fermiMotion=fermiMotion
    targetNuc%densitySwitch_static=densitySwitch_static

    targetNuc%radius_input = radius_input
    targetNuc%surface_input = surface_input

    targetNuc%doPrintGlauber = doPrintGlauber

    if (A <= 2) ReAdjustForConstBinding = .false.
    if (getRMF_flag()) ReAdjustForConstBinding = .false.
    if (eventtype == HeavyIon) ReAdjustForConstBinding = .false.

    if (ReAdjustForConstBinding) then
       write(*,*)
       write(*,*) 'We us the initial density distribution only'
       write(*,*) 'for getting some educated guess of the baryonic'
       write(*,*) 'potential. With this we get a density distribution'
       write(*,*) 'ensuring a constant binding energy. !!!!!!!!!!!'
       write(*,*) 'E_binding = ',ConstBinding

       if (ConstBinding>0.) then
          call Traceback('Binding energy must be negative. stop')
       end if
    end if

    targetNuc%ReAdjustForConstBinding = ReAdjustForConstBinding
    targetNuc%ConstBinding = ConstBinding

    call InitNucleus(targetNuc,fermiMomentum_input)

    targetNuc%DoInit = .false.

    call adjustTimesteps(targetNuc)

    write(*,*) 'doPrintGlauber = ',doPrintGlauber

    call Write_ReadingInput('target',2)

  end subroutine initTarget


  !****************************************************************************
  !****s* nucleus/initNucleus
  ! NAME
  ! subroutine initNucleus(Nuc, fermiMomentum_input)
  ! PURPOSE
  ! Initializes a nucleus resting in the frame of calculation at r=0.
  ! INPUTS
  ! * type(tNucleus), pointer :: Nuc
  ! * real, intent(in) :: fermiMomentum_input
  ! NOTES
  ! In 'Nuc', mass and charge have to be set as input variables.
  !
  ! we have to set nuc%radius, since this is referred to in further init
  ! routines (e.g. for calculating the distance)
  !****************************************************************************
  subroutine initNucleus(Nuc, fermiMomentum_input)
    use NucD, only: nuclfit
    use densityStatic, only: staticDensityInit
    use constants, only: pi, hbarc

    type(tNucleus), pointer :: Nuc
    real,intent(in) :: fermiMomentum_input
    real :: radius,surface,density,ratio

    if (Nuc%charge > Nuc%mass) then
      write(*,*) 'A, Z: ', Nuc%mass, Nuc%charge
      call Traceback('Wrong nucleus in the input.')
    end if

    select case (nuc%densitySwitch_static)
    case (5)
       if (Nuc%mass/=2*Nuc%charge) then
          write(*,*)
          write(*,*) 'WARNING: Using neutron-proton-symmetric Fermi-gas model'
          write(*,*) 'with non-symmetric nucleus! (in nucleus/initNucleus)'
          write(*,*) 'densitySwitch_static=', &
               nuc%densitySwitch_static, ', A=',Nuc%mass,' , Z=',Nuc%charge
          write(*,*)
       end if
       Nuc%surface=0.

       ratio = float(nuc%charge)/float(nuc%mass)
       Nuc%density(0) = 2*(fermiMomentum_input/hbarc)**3/(3.*pi**2)
       Nuc%density(1:2) = Nuc%density(0) * (/ ratio, 1.-ratio /)

       ! A=4/3 pi r^3 *density
       Nuc%radius=(float(nuc%mass)/(4./3.*pi*nuc%density(0)))**(1./3.)
       write(*,*) 'Fermi-gas parameters: '
       write(*,*) '# Radius of nucleus =' ,nuc%radius
       write(*,*) '# Density of nucleus=' ,nuc%density

    case default
       if (Nuc%mass>=12) then
          call nuclfit(Nuc%mass,Nuc%charge,radius,surface,density)
          Nuc%radius  = radius
          Nuc%surface = surface
          Nuc%density(0) = density
          Nuc%density(1:2) = density/2

       else
          write(*,*)
          write(*,*) 'WARNING: radius parameters not initialized !!!!!!!'
          write(*,*)
          ! the following is just dummy stuff. DO IT BETTER !!!
          if (Nuc%mass == 2) then
             Nuc%radius=2.
          else
             Nuc%radius=1.2*Nuc%mass**(1./3.)
          end if
          write(*,*) '...setting  Nuc%radius = ',Nuc%radius
       end if
    end select

    ! The above was just setting the default values.
    ! The input of Nuc%radius_input and Nuc%surface_input will be checked
    ! in staticDensityInit()

    call staticDensityInit(nuc)
    call nuc%writeParams()

  end subroutine initNucleus

  !****************************************************************************
  !****is* nucleus/adjustTimesteps
  ! NAME
  ! subroutine adjustTimesteps(Nuc)
  ! PURPOSE
  ! Adjust the overall running time of the simulation if the target is
  ! a helium target
  !****************************************************************************
  subroutine adjustTimesteps(Nuc)

    use inputGeneral, only: eventtype,variableTimeStep,numTimeSteps,time_max
    use eventTypes, only: Neutrino, HiPion, HiLepton

    type(tNucleus), pointer :: Nuc

    select case (eventType)
    case (Neutrino, HiPion, HiLepton)
       ! Set numTimeSteps = 0 for target mass A = 1  to avoid crash
       if (Nuc%mass == 1) then
          if (numTimeSteps>0) then
             numTimeSteps = 0
             if (.not.variableTimeStep) time_max = 0
             write(*,*)
             write(*,*) ' WARNING:',&
                  ' number of time steps set to 0 since target = H !'
             write(*,*)
          end if
       end if
    end select

  end subroutine adjustTimesteps


end module nucleus
