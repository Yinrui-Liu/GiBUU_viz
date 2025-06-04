!******************************************************************************
!****m* /nucleusDefinition
! NAME
! module nucleusDefinition
!
! PURPOSE
! Define structure "tNucleus".
!
! NOTES
! Because of internal module dependencies we do not include any routines
! working with/on the defined objects in this module but rather give them
! in the module "nucleus".
!******************************************************************************
module nucleusDefinition

  implicit none
  private

  integer, parameter :: maxIndex_def= 2000
  real,    parameter ::       dx_def=0.01

  !****************************************************************************
  !****t* nucleusDefinition/tNucleus
  ! SOURCE
  !
  type, public :: tNucleus
     real,dimension(0:2) :: radius = 0.  ! 0: baryon, 1: p, 2: n
     real,dimension(0:2) :: surface = 0. ! 0: baryon, 1: p, 2: n
     real,dimension(0:2) :: density = 0. ! 0: baryon, 1: p, 2: n
     real,dimension(1:3) :: pos = 0.  ! position in calculation frame
     real,dimension(1:3) :: vel = 0.  ! velocity in calculation frame
     integer   :: mass = 0            ! in units of nucleon masses
     integer   :: charge = 0
     logical   :: fermiMotion=.true.  ! switch to turn on/off FermiMotion

     logical   :: DoInit =.true.      ! if true, then first initialize !

     integer   :: densitySwitch_static = 3
     real      :: fermiMomentum_input = 0.225
     real,dimension(1:2) :: radius_input = -99.9
     real,dimension(1:2) :: surface_input = -99.9

     integer   :: MaxIndex = maxIndex_def
     real      :: dx       = dx_def
     real,dimension(0:maxIndex_def,0:2) :: densTab = 0. ! 0: baryon, 1: p, 2: n

     real      :: MaxDist = 0.   ! Distance-CutOff: dens < 1e-6
     real      :: MaxDens(2)=0.  ! maximum of densTab

     real      :: chemPot = 0.  ! 'chemical' pot. for LDA (relative units of E0)
     logical   :: ReAdjustForConstBinding = .false.
     real      :: ConstBinding = 0.
     real,dimension(2) :: fac = 0. ! scaling factors in Readjust

     logical   :: anti = .FALSE. ! true if antinucleus
     logical   :: doPrintGlauber = .false.


   contains
     procedure :: writeParams
     procedure :: writeStaticDens
     procedure :: writeAverageDens
     procedure :: staticDens
     procedure :: searchMaxVals
     procedure :: printGlauber
     procedure :: classicalEstimate

  end type tNucleus
  !
  ! PURPOSE
  ! This type stores informations about nuclei, namely target and projectile.
  !
  ! NOTES
  ! Following parameters are valid for all density parametrisations:
  ! * mass
  ! * charge
  !
  ! parameters for Woods-Saxon:
  ! * radius
  ! * surface
  ! * density
  !****************************************************************************

  ! This array is a local variable in classicalEstimate. It is made global
  ! to fix a bug in ifx 2025.0.4, when compiled with -ipo
  real, dimension(:,:), allocatable :: intRho

contains

  !****************************************************************************
  !****s* nucleusDefinition/writeParams
  ! NAME
  ! subroutine writeParams(this)
  ! PURPOSE
  ! write the main parameters to stdout
  ! INPUTS
  ! none
  ! OUTPUT
  ! written to stdout
  !****************************************************************************
  subroutine writeParams(this)
    class(tNucleus), intent(in) :: this

    write(*,'(A,i4)')    ' ::: Mass of nucleus    =',this%mass
    write(*,'(A,i4)')    ' ::: Charge of nucleus  =',this%charge
    write(*,'(A)')       ' :::                       Baryon:  Proton:  Neutron:'
    write(*,'(A,3f9.4)') ' ::: Radius of nucleus  =',this%radius
    write(*,'(A,3f9.4)') ' ::: Surface of nucleus =',this%surface
    write(*,'(A,3f9.4)') ' ::: Central density    =',this%density
  end subroutine writeParams


  !****************************************************************************
  !****s* nucleusDefinition/writeStaticDens
  ! NAME
  ! subroutine writeStaticDens(this, filename)
  ! PURPOSE
  ! write the static density to file
  !****************************************************************************
  subroutine writeStaticDens(this, filename)
    class(tNucleus), intent(in) :: this
    character*(*), intent(in) :: filename

    integer :: i
    write(*,*) 'writing static density to file: ',filename

    open(13,file=filename,status='unknown')
    write(13,'(A)') '# r [fm], rho_B, rho_p, rho_n'
    do i = 0, this%MaxIndex
       write(13,'(F6.2,3ES12.4)') i*this%dx, this%densTab(i,:)
    end do
    close(13)

  end subroutine writeStaticDens


  !****************************************************************************
  !****f* nucleusDefinition/staticDens
  ! NAME
  ! real function staticDens(this,r,iType)
  ! PURPOSE
  ! return value of the tabulated nuclear density
  ! INPUTS
  ! * real           :: r     -- radius [fm]
  ! * integer        :: iType -- 0: total, 1: proton, 2: neutron
  ! OUTPUT
  ! function value: density [fm^-3]
  !****************************************************************************
  pure real function staticDens(this, r, iType)

    class(tNucleus),intent(in) :: this
    real,           intent(in) :: r
    integer,        intent(in) :: iType

    integer :: i

    staticDens = -99.9

    if (iType<0) return
    if (iType>2) return

    i = nint(r/this%dx)
    if (i<0) return
    if (i>this%MaxIndex) return

    staticDens=this%densTab(i,iType)

  end function staticDens

  !****************************************************************************
  !****s* nucleusDefinition/writeAverageDens
  ! NAME
  ! subroutine writeAverageDens(this)
  ! PURPOSE
  ! Prints average density of nucleus
  !
  ! INPUTS
  ! none
  !
  ! NOTES
  ! <rho> = int(rho   rho r**2 dr)/int(rho r**2 dr)
  !****************************************************************************
  subroutine writeAverageDens(this)
    class(tNucleus),intent(in) :: this

    integer :: i
    real :: x, rho_average,rho_integral

    rho_average = 0.
    rho_integral= 0.

    do i=0,this%MaxIndex
       x = i*this%dx
       rho_average =rho_average +(this%densTab(i,0))**2 *  x**2
       rho_integral=rho_integral+(this%densTab(i,0))    *  x**2
    end do

    if (rho_integral/=0.) rho_average= rho_average/rho_integral

    write(*,'(A,F9.5)') ' ::: Average density of nucleus: ', rho_average

  end subroutine writeAverageDens

  !****************************************************************************
  !****s* nucleusDefinition/searchMaxVals
  ! NAME
  ! subroutine searchMaxVals(this)
  !
  ! PURPOSE
  ! go through the tabulated distributions to search for the extrema
  !****************************************************************************
  subroutine searchMaxVals(this)
    use constants, only:pi

    class(tNucleus),intent(inout) :: this

    integer :: i,j
    real :: x, maxV(2), maxX
    real :: Sum(0:3,0:2)

    maxV = 0.
    maxX = 0.
    Sum = 0.


    do i=0,this%MaxIndex
       x = i*this%dx
       if (this%densTab(i,1) > maxV(1)) maxV(1) = this%densTab(i,1)
       if (this%densTab(i,2) > maxV(2)) maxV(2) = this%densTab(i,2)
       if (this%densTab(i,0) > 1e-6) maxX = x

       do j=0,2
          Sum(j,:) = Sum(j,:)+x**(j+2)*this%densTab(i,:)
       end do
       Sum(3,:) = Sum(3,:)+x**2*this%densTab(i,:)**2
    end do

    Sum = Sum*this%dx

    this%MaxDist = maxX
    this%MaxDens = maxV

    write(*,*) 'Extrema for MC:'
    write(*,*) '    MaxDist = ',this%MaxDist
    write(*,*) '    MaxDens = ',this%MaxDens

    write(*,*) 'Integrations:'
    write(*,'("   ",A8,4A13)')  ' ', 'rho d^3r', 'rho r d^3r', 'rho r^2 d^3r', 'rho^2 d^3r'
    write(*,'("   ",A8,4f13.3)') 'Baryon:  ',Sum(0:3,0)*4.*pi
    write(*,'("   ",A8,4f13.3)') 'Proton:  ',Sum(0:3,1)*4.*pi
    write(*,'("   ",A8,4f13.3)') 'Neutron: ',Sum(0:3,2)*4.*pi
    write(*,*)

    if (this%radius(0) < 0.001) then
       this%radius = sqrt(Sum(2,:)/Sum(0,:))
       write(*,*) 'Attention! Setting radius to ',this%radius
    end if

  end subroutine searchMaxVals

  !****************************************************************************
  !****s* nucleusDefinition/printGlauber
  ! NAME
  ! subroutine printGlauber(this)
  ! PURPOSE
  ! This routine prints the total cross section sigma_hA according a Glauber
  ! calculation (cf. Falter PhD, eq. (5.15)) for different values of sigma_hN
  ! and a classical consideration.
  !****************************************************************************
  subroutine printGlauber(this)
    use constants, only: pi

    class(tNucleus), intent(in) :: this

    real, dimension(:,:), allocatable :: intRho
    integer :: ib,iz,ir,isigma
    real :: b,z,r,s_Gl,s_cl,sigma
    real, dimension(0:2) :: s2

    if (.not.this%doPrintGlauber) return

    ! 1) tabulate int_{-infty}^{+\infty} dz rho(b,z)

    allocate(intRho(0:this%MaxIndex,0:2))

    do ib=0,this%MaxIndex
       b = ib*this%dx
       s2 = 0
       do iz=0,this%MaxIndex
          z = iz*this%dx
          r = sqrt(b**2+z**2)
          ir = nint(r/this%dx)
          if (ir <= this%MaxIndex) &
               s2 = s2 + this%densTab(ir,:)
       end do
       intRho(ib,:) = 2*s2*this%dx ! *2 because of -infty..+infty
    end do

    ! 1a) print result to file
    open(13,file="Glauber0.dat",status='unknown')
    write(13,'(A)') "# b[fm], int dz rho_bar(b,z) [fm^-2], int dz rho_p(b,z) [fm^-2], int dz rho_n(b,z) [fm^-2]"
    do ib=0,this%MaxIndex
       b = ib*this%dx
       write(13,'(4ES12.4)') b,intRho(ib,:)
    end do
    close(13)

    ! 2) print table (only using rho_B)

    open(13,file="Glauber.dat",status='unknown')
    write(13,'(A)') "# sigma_hN[mb], sigma_hA^Glauber[mb], sigma_hA^classical[mb]"

    do isigma=1,200
       sigma = isigma / 10. ! in fm^2

       s_Gl = 0
       s_cl = 0
       do ib=0,this%MaxIndex
          b = ib*this%dx
          s_Gl = s_Gl + b*(1.-exp(-sigma/2*intRho(ib,0)))
          s_cl = s_cl + b*(1.-exp(-sigma*intRho(ib,0)))
       end do
       s_Gl = s_Gl * 2*this%dx * 2*pi
       s_cl = s_cl * this%dx * 2*pi
       write(13,'(3ES12.4)') sigma*10.,s_Gl*10.,s_cl*10. ! in mb
    end do
    close(13)

  end subroutine printGlauber

  !****************************************************************************
  !****f* nucleusDefinition/classicalEstimate
  ! NAME
  ! real function classicalEstimate(this, sigma)
  ! PURPOSE
  ! This routine calculates the total cross section sigma_hA according a
  ! classical picture for given values of sigma_hN
  !
  ! INPUTS
  ! * real, dimension(1:2) :: sigma -- sigmaP, sigmaN (in mb)
  !****************************************************************************
  real function classicalEstimate(this, sigma)
    use constants, only: pi

    class(tNucleus), intent(in) :: this
    real, dimension(1:2), intent(in) :: sigma

    ! This array is a local variable. It is made global
    ! to fix a bug in ifx 2025.0.4, when compiled with -ipo
    ! (error: malloc(): corrupted top size)
    !    real, dimension(:,:), allocatable :: intRho
    integer :: ib,iz,ir
    real :: b,z,r
    real, dimension(0:2) :: s2 = 0.

    ! 1) tabulate int_{-infty}^{+\infty} dz rho(b,z)

    allocate(intRho(0:this%MaxIndex,0:2))
    intRho = 0.

    do ib=0,this%MaxIndex
       b = ib*this%dx
       s2 = 0
       do iz=0,this%MaxIndex
          z = iz*this%dx
          r = sqrt(b**2+z**2)
          ir = nint(r/this%dx)
          if (ir <= this%MaxIndex) &
               s2 = s2 + this%densTab(ir,:)
       end do
       intRho(ib,:) = s2 * 2*this%dx ! *2 because of -infty..+infty
    end do

    ! 2) write the classical expression with input cross sections

    s2 = 0
    do ib=0,this%MaxIndex
       b = ib*this%dx
       s2(1:2) = s2(1:2) + b*(1.-exp(-sigma(1:2)/(2*10)*intRho(ib,1:2)))
    end do
    s2 = s2 * this%dx * 2*pi

    classicalEstimate = sum(s2(1:2))*10.  ! in mb

    deallocate(intRho)

  end function classicalEstimate


end module nucleusDefinition
