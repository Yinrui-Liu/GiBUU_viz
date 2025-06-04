!******************************************************************************
!****m* /initPion
! NAME
! module initPion
! PURPOSE
! Includes the initialization of pions for pion induced events.
!******************************************************************************
module initPion

  implicit none

  private

  public :: initPionInduced
  public :: getEkin
  public :: getTotalPerweight
  public :: getImpact


  !****************************************************************************
  !****g* pionNucleus/UseCoulomb
  ! SOURCE
  logical,save   :: UseCoulomb=.false.
  ! PURPOSE
  ! if .true. then a Coulomb propagation from CoulombDistance to distance is
  ! performed
  !****************************************************************************

  !****************************************************************************
  !****g* pionNucleus/CoulombDistance
  ! SOURCE
  real,save      :: CoulombDistance=200. ! [fm]
  ! PURPOSE
  ! distance from where the Coulomb propagation starts
  !****************************************************************************

  !****************************************************************************
  !****g* pionNucleus/distance
  ! SOURCE
  real,save      :: distance=15. ! [fm]
  ! PURPOSE
  ! initialization distance
  !****************************************************************************

  !****************************************************************************
  !****g* pionNucleus/impact_parameter
  ! SOURCE
  real,save      :: impact_parameter=0. ! [fm]
  ! PURPOSE
  ! impact parameter.
  ! If less than 0, than an impact parameter integration is performed
  !****************************************************************************

  !****************************************************************************
  !****g* pionNucleus/charge
  ! SOURCE
  integer,save   :: charge=0
  ! PURPOSE
  ! charge of pion
  !****************************************************************************

  !****************************************************************************
  !****g* pionNucleus/numberPions
  ! SOURCE
  integer,save   :: numberPions=200
  ! PURPOSE
  ! number of initialized pions per ensemble
  !****************************************************************************

  !****************************************************************************
  !****g* pionNucleus/ekin_lab
  ! SOURCE
  real,save      :: ekin_lab=0.  ! [GeV]
  !
  ! PURPOSE
  ! kinetic energies of pions in lab frame.
  !****************************************************************************

  !****************************************************************************
  !****g* pionNucleus/delta_ekin_lab
  ! SOURCE
  real,save      :: delta_ekin_lab=0.01  ! [GeV]
  ! PURPOSE
  ! step size for kinetic energies in energy scans
  !****************************************************************************

  !****************************************************************************
  !****g* pionNucleus/debug
  ! SOURCE
  logical, parameter :: debug=.false.
  ! PURPOSE
  ! Switch on debug mode
  !****************************************************************************

  real, save :: totalPerweight=0.

  real, save, allocatable :: bArr(:,:) ! array to store impact

  logical, save :: initFlag=.true.

contains

  !****************************************************************************
  !****f* initPion/getEkin
  ! NAME
  ! real function getEkin
  ! PURPOSE
  ! This function returns the kinetic energy of the pions as they were
  ! initialized at the last call of initPionInduced.
  !****************************************************************************
  real function getEkin()
    getEkin=ekin_lab
  end function getEkin


  !****************************************************************************
  !****f* initPion/getTotalPerweight
  ! NAME
  ! real function getTotalPerweight
  ! PURPOSE
  ! This function returns the total perweight of the pions as they were
  ! initialized at the last call of initPionInduced.
  !****************************************************************************
  real function getTotalPerweight()
    getTotalPerweight=totalPerweight
  end function getTotalPerweight

  !****************************************************************************
  !****f* initPion/getImpact
  ! NAME
  ! real function getImpact(firstevent)
  ! PURPOSE
  ! return the impact value connected with the pion inducing event indexed
  ! by firstevent
  !****************************************************************************
  real function getImpact(firstevent)
    use CallStack, only: traceback

    integer, intent(in) :: firstevent
    integer :: iPion,iEns, nEns

    if (initFlag) then
       call TRACEBACK("oops")
    end if

    nEns = size(bArr,dim=1)
    iPion = mod(firstevent,numberPions) ! firstevent=iPion+numberPions*(iEns-1)
    iEns = int(firstevent/numberPions)+1

    if (iEns<1.or.iEns>nEns) then
       write(*,*) firstevent,nEns,numberPions,iPion,iEns
       call TRACEBACK("oops j")
    end if
    if (iPion<1.or.iPion>numberPions) call TRACEBACK("oops iPion")

    getImpact = bArr(iEns,iPion)

  end function getImpact

  !****************************************************************************
  !****s* initPion/initPionInduced
  ! NAME
  ! subroutine initPionInduced(pertParts,raiseEnergyFlag,targetNuc)
  ! PURPOSE
  ! This routine initializes pions in pion-nucleus scattering.
  ! INPUTS
  ! * type(particle), intent(inout), dimension(:,:) :: pertParts
  !   -- vector to store pions in
  ! * logical, intent(in) :: raiseEnergyFlag
  !   -- if .true. energy of initialized pions is raised by delta_ekin_lab
  ! * type(tNucleus),pointer,optional :: targetNuc -- Target nucleus
  !****************************************************************************
  subroutine initPionInduced(pertParts,raiseEnergyFlag,targetNuc)
    use idTable
    use particleDefinition
    use nucleusDefinition
    use collisionNumbering, only: pert_Numbering ! Numbering of %event of the perturbative particles
    use output
    use residue, only: InitResidue

    type(particle), intent(inout), dimension(:,:), TARGET :: pertParts
    logical, intent(in) :: raiseEnergyFlag
    type(tNucleus),pointer,optional :: targetNuc

    integer :: iPart ! index of pion in vector pertParts
    integer :: iPion,iEns
    logical,save  :: outside
    logical :: successFlag
    integer :: nEns, nPart

    real :: impact
    type(particle), POINTER :: pPart

    nEns=size(pertParts,dim=1)
    nPart=size(pertParts,dim=2)

    write(*,*)
    write(*,subchapter) 'Initializing  pion induced events'

    if (initFlag) then
       ! Read input and check if pion is initialized in- or outside nucleus
       call initInput(outside)

!       write(*,*) 'allocating bArr(',nEns,',',numberPions,')'
       allocate(bArr(nEns,numberPions))
       bArr = 0.0

       initFlag=.false.
    end if

    if (raiseEnergyFlag) ekin_lab=ekin_lab+delta_ekin_lab
    write(*,*) 'Kinetic energy of pions in lab frame=', ekin_lab
    write(*,*) 'Outside-Flag=', outside
    totalPerweight=0.

    do iEns=1,nEns ! Loop over all Ensembles
       iPart=1
       iPion=0
       do
          do while(pertParts(iEns,iPart)%Id > 0) ! Find free place in particle vector
             iPart=iPart+1
             if (iPart.gt.nPart) then
                write(*,*) 'Particle vector too small in initPion'
                write(*,*) 'Size=',nPart
                write(*,*) 'Ensemble: ',iEns
                write(*,*) 'Number pions per ensemble=',numberPions
                stop
             end if
          end do

          pPart => pertParts(iEns,iPart)

          call setToDefault(pPart)
          call setKinematics
          call setPosition(impact)
          ! Give the particle its unique number
          call setNumber(pPart)

          ! correct kinematics if pion is initialized outside the nucleus:
          if (outside.and.UseCoulomb) call CoulombCorrect

          ! correct momentum if pion is initialized inside the nucleus:
          if (.not.outside) then
             call momentumCorrect(successFlag)
             if (.not.successFlag) then
                pPart%ID=0
                write(*,*) 'Generating event not succesful:',iEns,iPart
                cycle ! New Event!
             end if
          end if
          iPion=iPion+1
          ! please note: iPion may be different from iPart !!!!!
          pPart%firstevent=iPion+numberPions*(iEns-1)
          bArr(iEns,iPion) = impact ! store impact parameter for later access
          if (iPion.eq.numberpions) exit
       end do
    end do

    call InitResidue(nEns,numberPions,targetNuc%mass,targetNuc%charge)

    write(*,*) '**Finished Initializing pions for pion induced events'
    write(*,*) '**Total perweight=', totalPerweight
    write(*,*)

  contains

    !**************************************************************************
    !****s* initPionInduced/initInput
    ! NAME
    ! subroutine initInput(outside)
    ! PURPOSE
    ! Reads input out of jobcard. Namelist 'pionNucleus'.
    ! RESULT
    ! logical outside !whether pions are initialized in or outside the nucleus
    ! NOTES
    ! Checks wether pion is initialized in- or outside the nucleus
    !**************************************************************************
    subroutine initInput(outside)
      use output

      logical, intent(out) :: outside

      !************************************************************************
      !****n* initPion/pionNucleus
      ! NAME
      ! NAMELIST pionNucleus
      ! PURPOSE
      ! Includes parameters of pion initialization:
      ! * UseCoulomb
      ! * CoulombDistance
      ! * distance
      ! * impact_parameter
      ! * charge
      ! * numberPions
      ! * ekin_lab
      ! * delta_ekin_lab
      !************************************************************************

      NAMELIST /pionNucleus/ UseCoulomb,CoulombDistance,distance,&
           & impact_parameter,&
           & charge,numberPions,ekin_lab,delta_ekin_lab

      call Write_ReadingInput('pionNucleus',0)
      rewind(5)
      read(5,nml=pionNucleus)
      write(*,*) 'Impact Parameter                    =',impact_parameter
      write(*,*) 'Distance                            =',distance
      write(*,*) 'Coulomb correction for trajectories =',UseCoulomb
      write(*,*) 'Distance for the Coulomb correction =',CoulombDistance
      write(*,*) 'Kinetic Energy of pions in lab frame=',ekin_lab
      write(*,*) 'Delta(Energy) for energy scans      =',delta_ekin_lab
      write(*,*) 'Number of pions per ensemble        =',numberPions
      write(*,*) 'Charge of pions                     =',charge
      write(*,*)

      if (present(targetNuc)) then
         outside = (distance**2+impact_parameter**2.ge.(targetNuc%radius(0)+targetNuc%surface(0))**2)
      else
         outside=.false.
      end if

      if (outside) then
         write(*,*) 'Pions are initialized outside the nucleus'
      else
         write(*,*) 'Pions are initialized inside the nucleus'
      end if
      write(*,*)

      call Write_ReadingInput('pionNucleus',1)

    end subroutine initInput

    !**************************************************************************
    !****s* initPionInduced/setKinematics
    ! NAME
    ! subroutine setKinematics
    ! PURPOSE
    ! Sets basic kinematics of the pions.
    !**************************************************************************
    subroutine setKinematics

      use constants, only: mPi

      pPart%ID=pion
      pPart%charge=charge
      pPart%anti=.false.
      pPart%pert=.true.
      pPart%prodTime=0.
      pPart%mass=mPi
      pPart%mom(0)=ekin_lab+pPart%mass
      ! pion is initialized moving in positive z-direction:
      pPart%mom(1:3)=(/0.,0.,Sqrt(pPart%mom(0)**2-pPart%mass**2)/)
      ! assume vacuum dispersion relation:
      pPart%vel(1:3)=pPart%mom(1:3)/pPart%mom(0)
      pPart%event(1:2)=pert_numbering()
      if (debug) write(*,*) 'Masse=',pPart%mass

    end subroutine setKinematics

    !**************************************************************************
    !****s* initPion/setPosition
    ! NAME
    ! subroutine setPosition(impact)
    ! PURPOSE
    ! Sets positions of the pions.
    ! NOTES
    ! If Impact_Parameter is choosen to be less than zero, than the impact
    ! parameter is choosen by a Monte-Carlo-decision. This is made such that
    ! the pion is initialized on a disk of radius "bmax_Innerdisk" or on a
    ! ring which surrounds the inner disk and has an outer radius of
    ! "bmaxOuterRing".
    ! The probability to be on the inner disk is given by "pInnerDisk".
    ! The inner disk and the outer ring are separetely populated by a constant
    ! number density of pions.
    ! One distinguishes between inner disk and outer ring to have the
    ! possibility to have different
    ! population densities. Assumed one would only have one disk, then most of
    ! the particles would be
    ! initialized with high impact-parameter where only few reactions take
    ! place.
    !
    ! The perweight is given in units of mb for impact parameter integration.
    !
    ! OUTPUTS
    ! * pPart is changed
    ! * real :: impact -- the actual impact parameter for this pion
    !**************************************************************************
    subroutine setPosition(impact)

      use random
      use inputGeneral
      use constants

      real, intent(out) :: impact

      real :: bmax_OuterRing              !maximal Radius of outer ring
      real :: bmax_InnerDisk              !Radius of inner ring
      real, parameter :: pInnerDisk=0.7   !probability for initialization on inner ring
      real :: minimalDistance=2.52        !=pirp in old BUU
      !SQRT(maximal crossection of pion and nucleus/pi)
      real :: phi

      real, parameter :: ratioRadius=1.8 ! bmax_Outerring=ratioRadius*nuclearRadius+...

      integer :: totalNumPions
      real :: randomNumber
      logical, save :: flag = .true.

      totalNumPions=numberPions*nEns

      if (impact_parameter.ge.0.) then
         impact = impact_parameter
         pPart%pos=(/impact,0.,-distance/)

         !         if(fullensemble) then
         !            pPart%perweight=1./float(numberPions)
         !         else
         pPart%perweight=1./float(totalNumPions)
         !         end if
      else  ! Monte Carlo decision to have impact parameter integration
         ! maximum impact parameter of outer ring:
         if (fullEnsemble) then       ! Full ensemble
            minimalDistance=minimalDistance/sqrt(float(nEns))
         end if
         if (present(targetNuc)) then
            if (targetNuc%radius(0).gt.0.001) then  ! No elementary event
               bmax_OuterRing=ratioRadius*targetNuc%radius(0)+minimalDistance
               bmax_InnerDisk=targetNuc%radius(0)+minimaldistance
            else
               bmax_OuterRing=3.
               bmax_InnerDisk=2.
               if (fullEnsemble) then       ! Full ensemble
                  bmax_InnerDisk=bmax_InnerDisk  /sqrt(float(nEns))
                  bmax_OuterRing=bmax_OuterRing  /sqrt(float(nEns))
               end if
            end if
         else
            bmax_OuterRing=3.
            bmax_InnerDisk=2.
         end if
         if (flag) then
            write(*,*) '  Radius of outer ring:'  ,bmax_OuterRing
            write(*,*) '  Radius of inner circle:',bmax_InnerDisk
            write(*,*) '  perweight for pion in inner circle in fm^2:', (pi*bmax_InnerDisk**2)/pInnerDisk/float(totalNumPions)

            write(*,*) '  perweight for pion in outer ring in fm^2:', &
                 &        pi*(bmax_OuterRing**2-bmax_InnerDisk**2)/(1.-pInnerDisk)/float(totalNumPions)
            flag=.false.
         end if

         randomNumber=rn()
         phi=rn()*2*pi
         if (randomNumber.le.pInnerDisk) then
            impact=rn()
            pPart%pos(1)=sqrt(impact)*bmax_InnerDisk*cos(phi)
            pPart%pos(2)=sqrt(impact)*bmax_InnerDisk*sin(phi)
            pPart%pos(3)=-distance
            pPart%perweight=(pi*bmax_InnerDisk**2)/pInnerDisk /float(totalNumPions)*10 ! in mB (factor 10 due to fm**2 to mb conversion)

         else
            impact=rn()
            pPart%pos(1)=sqrt(impact*(bmax_OuterRing**2-bmax_InnerDisk**2)&
                 &                                      +bmax_InnerDisk**2)*cos(phi)
            pPart%pos(2)=sqrt(impact*(bmax_OuterRing**2-bmax_InnerDisk**2)&
                 &                                      +bmax_InnerDisk**2)*sin(phi)
            pPart%pos(3)=-distance

            pPart%perweight=pi*(bmax_OuterRing**2-bmax_InnerDisk**2)/(1.-pInnerDisk) &
                 &       /float(totalNumPions)*10  ! in mB (factor 10 due to fm**2 to mb conversion)
         end if
         !        if(fullensemble) pPart%perweight=pPart%perweight*float(nEns)
      end if
      totalPerweight=totalPerweight+pPart%perweight
    end subroutine setPosition

    !**************************************************************************
    !****s* initPion/CoulombCorrect
    ! NAME
    ! subroutine CoulombCorrect
    ! PURPOSE
    ! Corrects the trajectory according to Coulomb forces.
    !**************************************************************************
    subroutine CoulombCorrect
      use CoulombKorrektur, only: Coulpropa

      if (debug) then
         write(*,*) ' Before Coulomb correction of trajectory'
         write(*,*) 'position=', pPart%pos
         write(*,*) '4-momentum=', pPart%mom
      end if

      pPart%pos(3)=-coulombdistance

      call Coulpropa(pPart%pos(1:3), pPart%mom(1:3), &
           pPart%charge, pPart%mass, &
           targetNuc%charge, distance)

      !Assume vacuum dispersion relation:
      pPart%vel(1:3)=pPart%mom(1:3)/FreeEnergy(pPart)

      if (debug) then
         write(*,*) ' After Coulomb correction of trajectory'
         write(*,*) 'position=', pPart%pos
         write(*,*) '4-momentum=', pPart%mom
         write(*,*)
      end if
    end subroutine coulombCorrect

    !**************************************************************************

    subroutine momentumCorrect(successFlag)
      use coulomb, only: emfoca

      real :: cPot
      logical, intent(out) :: successFlag


      if (debug) then
         write(*,*) 'Correct for in medium potentials'
         write(*,*) 'Vacuum kinetic energy:', ekin_lab
      end if

      ! Evaluate coulomb potential and correct for coulomb potential:
      cpot=0.
      if (UseCoulomb) then
         cpot = emfoca(pPart%pos,pPart%mom(1:3),pPart%charge,pPart%ID)

         if (kineticEnergy(pPart)-cPot.lt.0) then
            write(*,*) "Error in Initpion"
            write(*,*) "Energy too small:",pPart%mom
            write(*,*) "Cannot initiliaze pions at",pPart%pos
            stop
         end if

         ! Correct momentum  p**2+m**2=(E-V_coulomb)**2 :
         pPart%mom(1:3)=(/0.,0.,SQRT((ekin_lab+pPart%mass&
              &                         -cPot)**2-pPart%mass**2)/)
         pPart%mom(0)=ekin_lab+pPart%mass-cpot

         if (debug) write(*,*) 'Coulomb potential:',cpot
      end if

      ! Correct for hadronic potential
      call RechneImpuls(pPart,ekin_Lab+pPart%mass-cpot, successFlag)
      if (debug) then
         write(*,*) "In medium kinetic energy:",kineticEnergy(pPart)
         write(*,'(A,4F9.6)') "In medium momentum:",pPart%mom
         write(*,*)
      end if

    end subroutine momentumCorrect

    !!*******************************************************

    subroutine RechneImpuls(partIn,energy,success)
      ! Newton-Routine um Impuls des Pions zu bestimmen
      ! Loese Gleichung Wurzel(m**2+p**2)+V(p_in_LRF)=energy
      use particleDefinition
      use potentialMain
      use energyCalc

      real, intent(in) :: energy
      type(particle),intent(inOut) :: partIn
      logical, intent(out) :: success
      !local
      integer, parameter :: maxSteps=100
      type(particle) :: part
      real, dimension(-1:1) ::f
      real, parameter :: dp=0.01
      real :: grad
      integer :: i,j


      part=partIn

      do j=0,maxSteps
         ! Evaluate derivative d(Energy_of_PartIn(p)-Energy)/dp
         do i=-1,1
            part%mom(1:2)=0.
            part%mom(3)=partIn%mom(3)+i*dp
            call energyDetermination(part)
            f(i)=part%mom(0)-energy
         end do
         if (abs(f(0)).lt.0.003) then
            if (debug) then
               write(*,*) 'Kinetic Energy in medium:',kineticEnergy(partIn)
               write(*,*) 'Kinetic Energy in vacuum:',ekin_lab
               write(*,*) 'Position=',partIn%pos
            end if
            success=.true.
            return
         end if
         if (debug) write(*,*) 'f=',f
         grad=(f(1)-f(-1))/2./dp
         if (abs(grad).lt.0.0001) then
            write(*,*) "Gradient zero in RechneImpuls von initPion", grad
            write(*,*) "Energy", energy
            write(*,*) "Momentum", part%mom
            write(*,*) "Step ", j
            write(*,*) f(-1),f(0),f(1)
            success=.false.
            return
         end if
         partIn%mom(1:2)=0.
         partIn%mom(3)=partIn%mom(3)-f(0)/grad
         call energyDetermination(part)
      end do
      write(*,*) "Fehler in initpion.f90,RechneImpuls",f(0)
      success=.false.
    end subroutine RechneImpuls

  end subroutine initpionInduced


end module initPion
