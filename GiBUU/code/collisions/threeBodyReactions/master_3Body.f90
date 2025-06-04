!******************************************************************************
!****m* /master_3Body
! NAME
! module master_3Body
! PURPOSE
! Includes all 3->X processes.
!******************************************************************************
module master_3Body

  use callstack, only: traceback

  implicit none
  private

  !****************************************************************************
  !****g* master_3Body/NNpion_NN_maxSrts
  ! PURPOSE
  ! Maximal sqrt(s) in GeV such that a NN Pion -> NN event takes place.
  ! SOURCE
  real, save :: NNpion_NN_maxSrts = 4.0
  !****************************************************************************

  !****************************************************************************
  !****g* master_3Body/positionNNpi
  ! PURPOSE
  ! This switch determines where the final state particles in NNpi->NN are
  ! positioned:
  ! * true: pion position
  ! * false: center of NNPi (default)
  ! SOURCE
  logical, save :: positionNNpi = .false.
  !****************************************************************************

  !****************************************************************************
  !****g* master_3Body/radiusNukSearch
  ! PURPOSE
  ! Radius for the search of nucleons, i.e. the radius in which nucleons
  ! shall be searched for at rho_0.
  ! SOURCE
  real, save :: radiusNukSearch = 2.9
  !****************************************************************************

  !****************************************************************************
  !****g* master_3Body/debug
  ! SOURCE
  logical, parameter :: debug=.false.
  ! PURPOSE
  ! Switch for debug information.
  !****************************************************************************

  !****************************************************************************
  !****g* master_3Body/correctEnergy
  ! PURPOSE
  ! Scale final state momenta to fulfill energy and momentum conservation.
  ! If .false., energy conservation is violated.
  ! SOURCE
  logical, save :: correctEnergy = .true.
  !****************************************************************************

  !****************************************************************************
  !****g* master_3Body/pionThreeBody
  ! PURPOSE
  ! Switch for the NNpion -> NN processes (false=OFF).
  ! SOURCE
  logical, save :: pionThreeBody = .true.
  !****************************************************************************

  !****************************************************************************
  !****g* master_3Body/DeltaThreeBody
  ! PURPOSE
  ! Switch for the NNDelta -> NNN processes (false=OFF).
  ! SOURCE
  logical, save :: DeltaThreeBody = .true.
  !****************************************************************************

  !****************************************************************************
  !****g* master_3Body/s_wave_abs
  ! PURPOSE
  ! Use the s-state absorption of Nieves et al NPA 554 (1993) 554, Eq. (18)
  ! for NNpion -> NN
  ! SOURCE
  logical, save :: s_wave_abs = .false.
  !
  ! NOTES
  ! * This switch should be .true. when the pion potential pionPot_Oset is used
  ! * with pionPot_Switch = 1 or = 4 in nl mesonPotential
  !****************************************************************************



  logical, save :: initFlag=.true.

  public :: make_3body_Collision
  public :: nukSearch
  public :: GetRadiusNukSearch

contains

  !****************************************************************************
  !****f* master_3Body/GetRadiusNukSearch
  ! NAME
  ! real function GetRadiusNukSearch()
  !
  ! PURPOSE
  ! return the value of the variable "radiusNukSearch"
  !****************************************************************************
  real function GetRadiusNukSearch()
    if (initFlag) call readInput
    GetRadiusNukSearch = radiusNukSearch
  end function GetRadiusNukSearch


  !****************************************************************************
  !****s* master_3Body/readInput
  ! NAME
  ! subroutine readInput
  ! PURPOSE
  ! Reads input in jobcard out of namelist "master_3body".
  !****************************************************************************
  subroutine readInput
    use output, only: Write_ReadingInput
    use masterPiDelta, only: getMasterPiDelta

    integer :: ios

    !**************************************************************************
    !****n* master_3Body/master_3body
    ! NAME
    ! NAMELIST /master_3body/
    ! PURPOSE
    ! Includes the switches:
    ! * correctEnergy
    ! * radiusNukSearch
    ! * DeltaThreeBody
    ! * pionThreeBody
    ! * positionNNpi
    ! * s_wave_abs
    !**************************************************************************
    NAMELIST /master_3Body/ correctEnergy,radiusNukSearch, &
         DeltaThreeBody,pionThreeBody,positionNNpi,s_wave_abs

    call Write_ReadingInput('master_3Body',0)

    rewind(5)
    read(5,nml=master_3Body,IOSTAT=ios)
    call Write_ReadingInput('master_3Body',0,ios)

    write(*,*) 'Correct energy by momentum scaling after collisions:', &
         correctEnergy
    write(*,*) 'Radius in which nucleons are searched for:', radiusNukSearch
    write(*,*) 'Delta N N -> N N N is included:', deltaThreeBody
    write(*,*) 'Pion N N  -> N N   is included:', pionThreeBody
    write(*,*) 'Pion N N special position description:', positionNNPi
    write(*,*) 'Pion N N Salcedo s-wave absorption:   ', s_wave_abs
    call Write_ReadingInput('master_3Body',1)

    initFlag=.false.

    if (getMasterPiDelta()) then
      s_wave_abs = .true.
      write (*,*) 'getMasterPiDelta: s_wave_abs set to .true. in master_3Body'
    end if

  end subroutine readInput

  !****************************************************************************
  !****s* master_3Body/make_3Body_Collision
  ! NAME
  ! subroutine make_3Body_Collision(PartIn,
  ! proton1,proton2,neutron1,neutron2,scatterPartner1,scatterPartner2,
  ! finalState,flagOK)
  !
  ! PURPOSE
  ! perform a 3-Body collision
  !
  ! INPUTS
  ! * type(particle) :: PartIn -- ...
  ! * type(particle), pointer :: proton1, proton2, neutron1, neutron2 -- ...
  !
  ! OUTPUT
  ! * type(particle), pointer :: scatterPartner1,scatterPartner2 -- ...
  ! * type(particle), dimension(:) :: finalState -- ...
  ! * logical :: flagOK -- ...
  !****************************************************************************
  subroutine make_3Body_Collision(PartIn, &
       proton1,proton2,neutron1,neutron2,scatterPartner1,scatterPartner2, &
       finalState,flagOK)

    use particleDefinition, only: particle,sqrts,useProductionPos,setProductionPos !,setTodefault
    use preEventDefinition
    use IdTable, only: pion, Delta
    use history, only: setHistory

    type(particle), intent(in) :: PartIn
    type(particle), pointer :: proton1, proton2, neutron1, neutron2, scatterPartner1, scatterPartner2
    type(particle), intent(out), dimension(:) :: finalState
    logical, intent(out) :: flagOK

    type(preEvent),dimension(1:3) :: preFinalState

    if (initFlag) call readInput

    if (debug) then
       write(*,*)
       write(*,'(A,3F8.3,2I4)') 'Particle:',PartIn%pos, PartIn%ID, PartIn%charge
       if (associated(proton1)) write(*,'(A,3F8.3,2I4)') &
            'Proton #1: ', proton1%pos,proton1%ID,proton1%charge
       if (associated(proton2)) write(*,'(A,3F8.3,2I4)') &
            'Proton #2: ',proton2%pos,proton1%ID,proton2%charge
       if (associated(neutron1)) write(*,'(A,3F8.3,2I4)') &
            'Neutron #1:',neutron1%pos,neutron1%ID,neutron1%charge
       if (associated(neutron2)) write(*,'(A,3F8.3,2I4)') &
            'Neutron #2:',neutron2%pos,neutron2%ID,neutron2%charge
    end if

    flagOK=.false.

    if (PartIn%ID.eq.pion.and.pionThreeBody) then
       call pionAbsorption(partIn,proton1,proton2,neutron1,neutron2, &
            scatterPartner1,scatterPartner2,preFinalState,flagOK)
    else if (PartIn%ID.eq.delta.and.deltaThreeBody) then
       call deltaAbsorption(partIn,proton1,proton2,neutron1,neutron2, &
            scatterPartner1,scatterPartner2,preFinalState,flagOK)
    end if

    if (.not.flagOK) return

    if (debug) then
       write(*,*) ScatterPartner1%ID,ScatterPartner1%Charge,ScatterPartner1%pos
       write(*,*) ScatterPartner2%ID,ScatterPartner2%Charge,ScatterPartner2%pos
    end if

    ! Initialize output
!    call setToDefault(finalState)

    call setFinalState(partIn,scatterPartner1,scatterPartner2,&
         preFinalState,finalState,flagOK)

    if (PartIn%ID.eq.pion.and.positionNNPI) then
       finalState%pos(1)=partIn%pos(1)
       finalState%pos(2)=partIn%pos(2)
       finalState%pos(3)=partIn%pos(3)
    end if

    if (debug) then
       write(*,*) 'flagOK', flagOK
       write(*,*) finalState%charge
       write(*,*) finalState%ID
       write(*,*) finalState%pert
       write(*,*) SQRTS(Finalstate(1),Finalstate(2),Finalstate(3))
       write(*,*) SQRTS(partIn,scatterPartner1,scatterPartner2)
       stop
    end if

    call setHistory(scatterPartner1, scatterPartner2, PartIn, finalState)

    if (useProductionPos) call setProductionPos(finalState)

  end subroutine make_3Body_Collision


  !****************************************************************************
  !****is* master_3Body/pionAbsorption
  ! NAME
  ! subroutine pionAbsorption(partIn,proton1,proton2,neutron1,neutron2, &
  ! scatterPartner1,scatterPartner2,preEvOut,flagOK)
  !
  ! PURPOSE
  ! perform a pi N N -> N N collision
  !
  ! INPUTS
  ! * type(particle) :: PartIn -- ...
  ! * type(particle), pointer :: proton1, proton2, neutron1, neutron2 -- ...
  !
  ! OUTPUT
  ! * type(particle), pointer :: scatterPartner1,scatterPartner2 -- ...
  ! * type(preEv), dimension(:) :: preEvOut -- ...
  ! * logical :: flagOK -- ...
  !****************************************************************************
  subroutine pionAbsorption(partIn,proton1,proton2,neutron1,neutron2, &
       scatterPartner1,scatterPartner2,preEvOut,flagOK)

    use NNPION_NN, only: gamma_NNPion_NN
    use densityModule, only: densityAt
    use dichteDefinition
    use IdTable, only: nucleon
    use random, only: rn
    use preEventDefinition
    use particleDefinition
    use inputGeneral, only: delta_T
    use XsectionRatios, only:  RatioNNPion_NN

    type(particle), intent(in) :: partIn
    type(particle), pointer :: proton1, proton2, neutron1, neutron2, &
         scatterPartner1, scatterPartner2
    type(preEvent), dimension(1:3), intent(out) :: preEvOut
    logical, intent(out) :: flagOK

    real :: Epion, rhoProton, rhoNeutron, srts, F, x, wahrscheinlichkeit, &
         gamma(0:2), eNucleon(1:2)
    integer, dimension(1:2) :: chargeNucleon
    type(dichte) :: density

    ! Initialize output
    preEvOut%Id=0
    preEvOut%charge=0
    flagOK=.false.

    density=densityAt(PartIn%pos)
    rhoProton=density%proton(0)
    rhoNeutron=density%neutron(0)

    gamma=0.

    Epion=FreeEnergy(PartIn)

    !**************************************************************************
    ! Set the absorption rates
    !**************************************************************************

    ! Set NN channel
    if (associated(neutron1).and.(Associated(neutron2))) then
       chargeNucleon=(/0,0/)
       Enucleon(1)=FreeEnergy(neutron1)
       Enucleon(2)=FreeEnergy(neutron2)
       srts=sqrt((Sum(Enucleon)+Epion)**2 &
            -Dot_Product(PartIn%mom(1:3)+neutron1%mom(1:3)+neutron2%mom(1:3), &
            &            PartIn%mom(1:3)+neutron1%mom(1:3)+neutron2%mom(1:3)))
       if (srts.gt.NNPION_NN_maxSrts) then
          if (debug) write(*,*) 'srts=',srts,'>',NNPION_NN_maxSrts,&
               '=NNPION_NN_maxSrts. No absorption'
       else
          gamma(0)=gamma_NNPion_NN(srts,Epion,Enucleon,rhoProton,rhoNeutron,PartIn%charge,chargeNucleon,s_wave_abs)
          if (debug) write(*,*) 'Energies:',Epion,Enucleon, 'Srts=',srts
          F=RatioNNPion_NN(neutron1,neutron2,PartIn) ! In-medium correction
          gamma(0)=gamma(0)*F
       end if
    end if

    ! Set NP channel
    if (associated(neutron1).and.(Associated(proton1))) then
       chargeNucleon=(/0,1/)
       Enucleon(1)=FreeEnergy(neutron1)
       Enucleon(2)=FreeEnergy(proton1)
       srts=sqrt((Sum(enucleon)+Epion)**2 &
            -Dot_Product(PartIn%mom(1:3)+neutron1%mom(1:3)+proton1%mom(1:3), &
            &            PartIn%mom(1:3)+neutron1%mom(1:3)+proton1%mom(1:3)))
       if (srts.gt.NNPION_NN_maxSrts) then
          if (debug) write(*,*) 'srts=',srts,'>',NNPION_NN_maxSrts, &
               '=NNPION_NN_maxSrts. No absorption'
       else
          gamma(1)=gamma_NNPion_NN(srts,Epion,Enucleon,rhoProton,rhoNeutron,PartIn%charge,chargeNucleon,s_wave_abs)
          if (debug) write(*,*) 'Energies:',Epion,Enucleon, 'Srts=',srts
          F=RatioNNPion_NN(neutron1,proton1,PartIn)  ! In-medium correction
          gamma(1)=gamma(1)*F
       end if
    end if

    ! Set PP channel
    if (associated(proton1).and.(Associated(proton2))) then
       chargeNucleon=(/1,1/)
       Enucleon(1)=FreeEnergy(proton1)
       Enucleon(2)=FreeEnergy(proton2)
       srts=sqrt((Sum(enucleon)+Epion)**2 &
            -Dot_Product(PartIn%mom(1:3)+proton1%mom(1:3)+proton2%mom(1:3), &
            &            PartIn%mom(1:3)+proton1%mom(1:3)+proton2%mom(1:3)))
       if (srts.gt.NNPION_NN_maxSrts) then
          if (debug) write(*,*) 'srts=',srts,'>',NNPION_NN_maxSrts,&
               '=NNPION_NN_maxSrts. No absorption'
       else
          gamma(2)=gamma_NNPion_NN(srts,Epion,Enucleon,rhoProton,rhoNeutron,PartIn%charge,chargeNucleon,s_wave_abs)
          if (debug) write(*,*) 'Energies:',Epion,Enucleon, 'Srts=',srts
          F=RatioNNPion_NN(proton1,proton2,PartIn)  ! In-medium correction
          gamma(2)=gamma(2)*F
       end if
    end if


    !**************************************************************************
    ! Decide whether absorption event takes place:
    !**************************************************************************

    !Monte-Carlo decision if particle is going to be absorbed
    if (debug)  write(*,*) 'wahrsch:',delta_T,gamma,Sum(gamma)

!    write(*,*) 'wahrsch:',delta_T,gamma,Sum(gamma)

!    if (Sum(gamma).gt.1e10) then
!       wahrscheinlichkeit=0.0
!    else
       wahrscheinlichkeit=exp(-delta_T*Sum(gamma)/0.197)
!    endif
    !dt anstatt dt0, da Zerfallsbreite im Laborsystem berechnet wurde

    if (debug) write(*,*) wahrscheinlichkeit
    if (debug) write(*,*) "in auswuerfeln",gamma,sum(gamma)

    x=rn()

    if (x.le.wahrscheinlichkeit) then !No absorption
       if (debug) write(*,*) x,'<',wahrscheinlichkeit,'No absorption'
       return
    end if

    if (debug) write(*,*) x,'>',wahrscheinlichkeit,'Absorption'

    !**************************************************************************
    ! Choose channel
    !**************************************************************************
    preEvOut(1:2)%ID=nucleon

    x=rn()*Sum(gamma)

    if (x.lt.gamma(0)) then
       ! NN Pion  -> NN o NP
       preEvOut(1:2)%charge=(/PartIn%charge,0/)
       scatterPartner1=>neutron1
       scatterPartner2=>neutron2
       if (associated(proton1)) nullify(proton1)
       if (associated(proton2)) nullify(proton2)

    else if (x.lt.gamma(0)+gamma(1)) then
       ! NP  channel -> NN or NP or PP
       select case (PartIn%charge)
       case (1)
          preEvOut(1:2)%charge=(/1,1/)
       case (0)
          preEvOut(1:2)%charge=(/1,0/)
       case (-1)
          preEvOut(1:2)%charge=(/0,0/)
       end select
       scatterPartner1=>proton1
       scatterPartner2=>neutron1
       if (associated(proton2)) nullify(proton2)
       if (associated(neutron2)) nullify(neutron2)

    else
       ! PP  channel->PP or PN
       preEvOut(1:2)%charge=(/1,1+PartIn%charge/)
       scatterPartner1=>proton1
       scatterPartner2=>proton2
       if (associated(neutron1)) nullify(neutron1)
       if (associated(neutron2)) nullify(neutron2)
    end if

    flagOK=.true.
    if (debug) write(*,*) 'Charge=',preEvOut%charge, 'Id=',preEvOut%ID

  end subroutine pionAbsorption


  !****************************************************************************
  !****is* master_3Body/deltaAbsorption
  ! NAME
  ! subroutine deltaAbsorption(partIn,proton1,proton2,neutron1,neutron2, &
  ! scatterPartner1,scatterPartner2,preEvOut,flagOK)
  !
  ! PURPOSE
  ! perform a Delta N N -> N N N collision
  !
  ! INPUTS
  ! * type(particle) :: PartIn -- ...
  ! * type(particle), pointer :: proton1, proton2, neutron1, neutron2 -- ...
  !
  ! OUTPUT
  ! * type(particle), pointer :: scatterPartner1,scatterPartner2 -- ...
  ! * type(preEv), dimension(:) :: preEvOut -- ...
  ! * logical :: flagOK -- ...
  !****************************************************************************
  subroutine deltaAbsorption(partIn,proton1,proton2,neutron1,neutron2, &
       scatterPartner1,scatterPartner2,preEvOut,flagOK)

    use densityModule, only: densityAt
    use dichteDefinition
    use IdTable, only: nucleon, Delta
    use random, only: rn
    use preEventDefinition
    use particleDefinition
    use inputGeneral, only: delta_T
    use baryonWidthMedium, only: get_MediumSwitch_Delta
    use deltaWidth, only: delta_fullWidth, delta_nucleonPion, deloset
    use barbar_barbar, only: delta2Body_inMedium_treatment

    type(particle), intent(in) :: partIn
    type(particle), pointer :: proton1, proton2, neutron1, neutron2, &
         scatterPartner1, scatterPartner2
    type(preEvent), dimension(1:3), intent(out) :: preEvOut
    logical, intent(out) :: flagOK

    !real :: rhoProton, rhoNeutron
    type(dichte) :: density
    real :: gamma, mass, dens, gammaLorentz, momentum(1:3), x, &
         wahrscheinlichkeit
    logical, dimension(0:3) :: channelFlag
    integer :: numChannels,i
    logical, parameter :: debugDelta=.false.
    real :: imsig2,imsig3,imsigq

    ! Initialize output
    preEvOut%Id=0
    preEvOut%charge=0
    flagOK=.false.

    density=densityAt(PartIn%pos)
    !rhoProton=density%proton(0)
    !rhoNeutron=density%neutron(0)

    if (partIn%ID.ne. Delta) then
       write(*,*) 'ID=', partIn%ID
       call Traceback("Wrong input in deltaAbsorption: No Delta!")
    end if

    if (partIn%anti) then
       return
    end if

    !**************************************************************************
    ! Set the absorption rates
    !**************************************************************************

    if (get_MediumSwitch_Delta()) then
       mass=PartIn%Mass
       dens=density%proton(0)+density%neutron(0)
       momentum=partIn%mom(1:3)
       if (delta2Body_inMedium_treatment()) then
          ! QE and 2-body contribution have already been considered in 2-body
          ! processes
          ! -> Include only 3-body contribution!!
          call deloset(mass,dens,imsig2,imsig3,imsigq)
          gamma = abs(2.*imsig3)
       else
          gamma = delta_fullWidth(mass,momentum,dens) &
               - delta_nucleonPion(mass,momentum,dens)
       end if
    else
       gamma = 0.
       return
    end if

    gammaLorentz=FreeEnergy(partIn)/partIn%mass

    !**************************************************************************
    ! Decide whether absorption event takes place:
    !**************************************************************************

    !Monte-Carlo decision if particle is going to be absorbed
    wahrscheinlichkeit=exp(-delta_T/gammaLorentz*gamma/0.197)
    !dt0 anstatt dt, da Zerfallsbreite im Ruhesystem berechnet

    if (debug) write(*,*) wahrscheinlichkeit
    if (debug) write(*,*) "in auswuerfeln",gamma

    x=rn()

    if (x.le.wahrscheinlichkeit) then !No absorption
       if (debug) write(*,*) x,'<',wahrscheinlichkeit,'No absorption in delta'
       return
    end if

    if (debug) write(*,*) x,'>',wahrscheinlichkeit,'Absorption in delta'
!    flagOK=.true.

    !**************************************************************************
    ! Pick resulting nucleons, to conserve energy
    !**************************************************************************

    numChannels=0 ! number of open channels

    ! We denote :
    ! "channelFlag(i)" will show whether channel "i" is open
    ! "i" is the sum of nucleon charges in the incoming channel:
    ! 0 : neutron neutron
    ! 1 : neutron proton
    ! 2 : proton proton
    channelFlag(0:2)=.false.  ! Initialization

    select case (partIn%charge)
    case (-1)
       channelFlag(0)=.false.  ! no absorption on nn possible
       channelFlag(1)=(associated(neutron1).and.associated(proton1))
       channelFlag(2)=(associated(proton1).and.associated(proton2))
    case (0)
       channelFlag(0)=(associated(neutron1).and.associated(neutron2))
       channelFlag(1)=(associated(neutron1).and.associated(proton1))
       channelFlag(2)=(associated(proton1).and.associated(proton2))
    case (1)
       channelFlag(0)=(associated(neutron1).and.associated(neutron2))
       channelFlag(1)=(associated(neutron1).and.associated(proton1))
       channelFlag(2)=(associated(proton1).and.associated(proton2))
    case (2)
       channelFlag(0)=(associated(neutron1).and.associated(neutron2))
       channelFlag(1)=(associated(neutron1).and.associated(proton1))
       channelFlag(2)=.false.  ! no absorption on pp possible

    case default
       write(*,*) 'Strange charge in delta absorption', &
            partIn%charge,partIn%anti,partIn%ID,partIn%mom
       call Traceback()
    end select

    numChannels = count(channelFlag)
    if (numChannels==0) then
       flagOK=.false.
       return
    end if

    ! Decide for channel, loop over all possible channels
    x=rn()
    wahrscheinlichkeit=0.

    DECIDE_loop : do i=0,2
       if (channelFlag(i)) then ! CHANNEL IS OPEN
          wahrscheinlichkeit=wahrscheinlichkeit+float(1)/float(numChannels)
       end if
       if ((x.le.wahrscheinlichkeit).and.channelFlag(i)) then
          ! found channel
          select case (i)
          case (0)
             scatterPartner1=>neutron1
             scatterPartner2=>neutron2
             if (associated(proton1)) nullify(proton1)
             if (associated(proton2)) nullify(proton2)
             select case (PartIn%charge)
             case (0)
                preEvOut(1:3)%Charge=(/0,0,0/)
             case (1)
                preEvOut(1:3)%Charge=(/0,0,1/)
             case (2)
                preEvOut(1:3)%Charge=(/0,1,1/)
             case default
                write(*,*) 'error in deltaAbsorption A ',i,partIn%Charge
                call Traceback()
             end select
          case (1)
             scatterPartner1=>neutron1
             scatterPartner2=>proton1
             if (associated(neutron2)) nullify(neutron2)
             if (associated(proton2)) nullify(proton2)
             select case (PartIn%charge)
             case (-1)
                preEvOut(1:3)%Charge=(/0,0,0/)
             case (0)
                preEvOut(1:3)%Charge=(/0,0,1/)
             case (1)
                preEvOut(1:3)%Charge=(/0,1,1/)
             case (2)
                preEvOut(1:3)%Charge=(/1,1,1/)
             case default
                write(*,*) 'error in deltaAbsorption B ',i,partIn%Charge
                call Traceback()
             end select
          case (2)
             scatterPartner1=>proton2
             scatterPartner2=>proton1
             if (associated(neutron2)) nullify(neutron2)
             if (associated(neutron1)) nullify(neutron1)
             select case (PartIn%charge)
             case (-1)
                preEvOut(1:3)%Charge=(/1,0,0/)
             case (0)
                preEvOut(1:3)%Charge=(/1,1,0/)
             case (1)
                preEvOut(1:3)%Charge=(/1,1,1/)
             case default
                write(*,*) 'error in deltaAbsorption C ',i,partIn%Charge
                call Traceback()
             end select
          end select
          preEvOut(1:3)%ID=nucleon ! nucleons are resulting particles
          flagOK=.true.
          exit DECIDE_loop
       end if
    end do DECIDE_loop
    ! Check the outcome
    if (.not.(associated(scatterPartner1).and.associated(scatterPartner2))) then
       write(*,*) 'error in deltaAbsorption',&
            associated(scatterPartner1),associated(scatterPartner2), &
            associated(neutron1),associated(neutron2), &
            associated(proton1),associated(proton2), &
            numChannels,x, wahrscheinlichkeit
       call Traceback()
    else
       flagOK=.true.
    end if
    if (debugDelta) then
       write(*,*) 'deltaAbs:',preEvOut%ID, preEvOut%Charge, &
            partIn%ID,partIn%charge
       write(*,*) associated(scatterPartner1),associated(scatterPartner2), &
            '####' ,associated(neutron1),associated(neutron2), &
            '####',associated(proton1),associated(proton2), &
            'num Channels', numChannels,'random:',x, wahrscheinlichkeit
    end if
  end subroutine deltaAbsorption

  !****************************************************************************
  !****is* master_3Body/setFinalState
  ! NAME
  ! subroutine setFinalState(part1,part2,part3,preFinalState,finalState,flagOK)
  !
  ! PURPOSE
  ! ...
  !
  ! INPUTS
  ! ...
  !
  ! OUTPUT
  ! ...
  !****************************************************************************
  subroutine setFinalState(part1,part2,part3,preFinalState,finalState,flagOK)

    use preEventDefinition
    use particleDefinition, only: particle, setToDefault, freeEnergy
    use dichteDefinition
    use MediumDefinition
    use lorentzTrafo, only: lorentz, lorentzCalcBeta
    use propagation, only: updateVelocity, checkVelo
    use densityModule, only: densityAt,energyDeterminationRMF
    use offShellPotential, only: setOffShellParameter
    use mediumModule, only: mediumAt,getMediumCutOff
    use energyCalc, only: energyDetermination
    use RMF, only: getRMF_flag

    type(particle), intent(in) :: part1,part2,part3
    type(preEvent), intent(in),dimension(:) :: preFinalState
    type(particle), dimension(:), intent(out) :: finalState
    logical, intent(out) :: flagOK

    real, dimension(0:3) :: mom_calc,mom_LRF
    type(dichte) :: density
    type(medium) :: mediumAtColl
    real, dimension(1:3) :: betaToLRF,betaToCM,position
    integer :: maxID, k
    real :: sqrts, sqrts_vac
    logical :: success

    finalState%Id=0
    ! (1) Evaluate Sqrt(s) using the calculation frame
    mom_calc(0:3)=part1%mom(0:3)+part2%mom(0:3)+part3%mom(0:3)
    sqrtS=sqrt(mom_calc(0)**2-Dot_Product(mom_calc(1:3),mom_calc(1:3)))

    ! (2) Define Sqrt(s) in the vacuum
    sqrtS_vac=SQRT((FreeEnergy(part1)+FreeEnergy(part2)+FreeEnergy(part3))**2 &
                       & -Dot_Product(mom_calc(1:3),mom_calc(1:3)))
    if (sqrtS_vac.lt.part1%mass+part2%mass+part3%mass) then
       write(*,*) 'Error in master_3Body: sqrtS_vac.lt.Sum(masses): sqrtS_vac'
       write(*,*) 'stopping'
       stop
    end if
    ! (3) Read out medium at collision point in LRF
    position=(part1%pos+part2%pos+part3%pos)/3.
    density=densityAt(position)
    mediumAtColl=mediumAt(density,position)

    ! (4) Define total momentum in LRF. If density not negligible
    ! then boost is needed.
    mom_LRF=mom_Calc
    if (density%baryon(0).gt.getMediumCutOff()/100.) then
       betaToLRF = lorentzCalcBeta(density%baryon)
       call lorentz(betaToLRF, mom_LRF)
    else
       betaToLRF=0.
    end if

    if (debug) write(*,*) sqrtS_vac,mediumAtColl%useMedium, mom_LRF

    ! (4a) Intialize output
    call setToDefault(finalState)
    if (.not.part1%pert.and.(.not.part2%pert).and.(.not.part3%pert)) then
       finalstate%pert=.false.
    else
       finalState%pert=.true.
    end if

    finalState(1:3)%ID=preFinalState(1:3)%ID
    finalState(1:3)%charge=preFinalState(1:3)%charge
    finalState(1:3)%anti=preFinalState(1:3)%anti
    finalState(1:3)%mass=preFinalState(1:3)%mass

    maxId=0
    do k=lbound(finalState,dim=1),ubound(finalState,dim=1)
       finalState(k)%pos=position
       if (finalState(k)%Id.eq.0) then
          maxID=k-1
          exit
       else if (k.eq.ubound(finalState,dim=1)) then
          maxID=ubound(finalState,dim=1)
          exit
       end if
    end do

    ! (10) Set kinematics of outgoing particles
    betaToCM = lorentzCalcBeta(mom_calc)
    call setKinematics(sqrtS,sqrtS_vac,mom_calc, betaToLRF,betaToCM, &
         mediumAtColl,finalState(1:maxID),flagOK)
    if (.not.flagOK) return

    ! (11) Update velocities
    do k=lbound(finalState,dim=1),ubound(finalState,dim=1)
       if (finalstate(k)%id.eq.0) cycle
       if(.not.getRMF_flag()) then
          call energyDetermination(finalstate(k))
       else
          call energyDeterminationRMF(finalstate(k))
       end if
       call setOffShellParameter(finalstate(k),success)
       if (.not.success) then
          flagOK = .false.
          return
       end if
       if(.not.getRMF_flag() .or. abs(finalstate(k)%offshellPar).gt.1.E-06) then
          call updateVelocity(finalState(k),success)
       else
          finalState(k)%vel(1:3)=finalState(k)%mom(1:3)/finalState(k)%mom(0)
          success=checkVelo(finalState(k))
       end if
       if (.not.success) then
          write(*,*) 'MasterThreeBody: velocity ge 1: collisionFlag -> FALSE'
          flagOK = .false.
          return
       end if
    end do

  end subroutine setFinalState


  !****************************************************************************
  !****s* master_3Body/setKinematics
  ! NAME
  ! subroutine setKinematics(srts,srts_vac,mom_calc, betaToLRF,betaToCM,medium_AtCollision, finalState)
  ! PURPOSE
  ! Evaluates the kinematics for the "finalState" particles assuming vacuum kinematics (-> no potentials).
  ! INPUTS
  ! * real                 :: srts       -- sqrt(s)
  ! * real                 :: srts_vac   -- sqrt(s) in the vacuum
  ! * real, dimension(0:3) :: mom_calc   -- total momentum in calculation frame
  ! * real, dimension(1:3) :: betaToLRF  -- beta of calc frame to LRF frame
  ! * real, dimension(1:3) :: betaToCM   -- beta of cal frame to CM frame
  ! * type(medium)         :: mediumAtCollision -- medium information
  ! OUTPUT
  ! * type(particle), dimension(:) :: finalState -- ...
  ! * logical :: collisionFlag -- "true" if kinematics could be set, "false" if not
  ! NOTES
  ! It's important that the Id's and charges of the "finalState" particles are
  ! already set when calling this subroutine.
  ! Only kinematics including masses of this finalState will be set.
  !****************************************************************************
  subroutine setKinematics(srts,srts_vac,mom_calc,betaToLRF,betaToCM, &
       medium_AtCollision,finalState,collisionFlag)

    use mediumDefinition
    use particleDefinition
    use finalStateModule, only: assMass, massAss
    use energyCalc, only: energyCorrection
    use lorentzTrafo, only: lorentz
    use output, only: WriteParticle
    use propagation, only: checkVelo

    real, intent(in)                            :: srts, srts_vac
    real, intent(in), dimension(0:3)            :: mom_calc
    real, intent(in), dimension(1:3)            :: betaToLRF, betaToCM
    type(medium), intent(in)                    :: medium_AtCollision
    type(particle), intent(inOut), dimension(:) :: finalState
    logical, intent(out)                        :: collisionFlag

    integer :: i,j
    real, parameter, dimension (1:3) :: spotOut3 = 0.
    logical :: flag, flagOK
    integer, parameter :: maxCorrectLoop=10     ! maximal number of iterations for energy correction
    real :: impuls(0:3), betaDummy(1:3)
    type(particle), dimension (1:2)  :: pairInDummy
    integer, save :: fehlerZaehler=0,fehlerZaehler2=0

    pairInDummy%ID=(/0,0/)

    collisionFlag=.true.

    ! CHECK INPUT
    do i=lbound(finalState,dim=1),ubound(finalState,dim=1)
       if (finalState(i)%Id.eq.0) then
          write(*,*) 'Critical Error in setKinematics'
          write(*,*) 'Particle ID equals 0!!!'
          write(*,*) 'stop'
       end if
    end do

    flag=.true.

    ! Set Energy and momenta

    select case (size(finalState,dim=1))
    case (1)
       ! Resonance Production
       finalState(1)%mom=mom_calc
    case (2)
       energyCorrectLoop : do i=1, maxCorrectLoop
          ! two body final state.
          if (debug) write(*,*) 'Vor massAss'
          call massass(srts_vac,medium_AtCollision,pairInDummy,finalState,betaToLRF,betaToCM,0,flag)
          if (.not.flag) then
             write(*,*) 'master_3Body: Impossible to find final state (1)'
             write(*,*) finalState%ID
             write(*,*) srts, srts_vac

             call WriteParticle(6,99,1,pairInDummy(1))
             call WriteParticle(6,99,2,pairInDummy(2))

             call WriteParticle(6,-99,1,finalState(1))
             call WriteParticle(6,-99,2,finalState(2))

             fehlerZaehler=fehlerzaehler+1
             if (fehlerZaehler.gt.50000) then
                call Traceback('>50000 errors in master_3Body: massass -> STOP')
             end if

            collisionFlag=.false.
            ! Kill Event
            finalState(1:2)%ID=0
            return ! --> Exit
          end if

          if (correctEnergy) then
             if (debug) write(*,*) '*************************'
             if (debug) write(*,*) 'wished srts=', srts
             call energyCorrection(srts,betaToCM, finalState, flagOK)
             if (debug) write(*,*) 'final srts=', sqrts(finalState(1),finalState(2))
             if (flagOK) exit energyCorrectLoop
          else
             ! just boost to Calculation frame
             flagOK=.true.
             do j=1,2
                impuls=finalState(j)%mom
                betaDummy=-betaToCM
                call lorentz(betaDummy,impuls)
                finalState(j)%mom=impuls
             end do
             exit energyCorrectLoop
          end if
       end do energyCorrectLoop

       if (.not.flagOK) then
          write(*,*) 'Energy correction failed in setKinematics of three body interactions'
          ! Kill Event
          collisionFlag=.false.
          finalState(1:2)%ID=0
          return
       end if

    case (3)
       do i=1, maxCorrectLoop
          ! three body final state. Neglecting possible scalar potentials, since spotOut=0
          call assMass(srts_vac,medium_AtCollision,pairInDummy,finalState,&
               spotOut3,betaToLRF,betaToCM,flag)
          if (.not.flag) then
             write(*,*) 'master_3Body: Impossible to find final state (2)', &
                  finalState%ID
             write(*,*) finalState%ID
             write(*,*) srts, srts_vac

             call WriteParticle(6,99,1,pairInDummy(1))
             call WriteParticle(6,99,2,pairInDummy(2))

             call WriteParticle(6,-99,1,finalState(1))
             call WriteParticle(6,-99,2,finalState(2))
             call WriteParticle(6,-99,3,finalState(3))

             fehlerZaehler2=fehlerzaehler2+1
             if (fehlerZaehler2.gt.50000) then
                call Traceback('>50000 errors in master_3Body: assmass -> STOP')
             end if

             collisionFlag=.false.
             ! Kill Event
             finalState(1:3)%ID=0
             return ! --> Exit
          end if

          if (correctEnergy) then
             call energyCorrection(srts,betaToCM, finalState, flagOK)
             if (flagOK) exit
          else
             ! just boost to Calculation frame
             flagOK=.true.
             do j=1,3
                impuls=finalState(j)%mom
                call lorentz(-betaToCM,impuls)
                finalState(j)%mom=impuls
             end do
          end if
       end do

       if (.not.flagOK) then
          write(*,*) 'Energy correction failed in setKinematics'
          collisionFlag=.false.
          ! Kill Event
          finalState(1:3)%ID=0
          write(*,*) 'setkinematics, energy correction loop, 3 particle final state failed of three-body-interaction'
          return
       end if

    case default
       write(*,*) 'Error in setKinematics: No treatment of more than' &
            &, ' a three-particle-final state implemented yet!'
       write(*,*) size(finalState), finalState(:)%ID
       stop
    end select

    ! set velocities in vacuum approximation

    do i=lbound(finalState,dim=1), ubound(finalState, dim=1)
       finalstate(i)%vel=finalState(i)%mom(1:3)/finalState(i)%mom(0)
       if (.not.checkVelo(finalState(i))) then
          write(*,*) 'checkVelo failed in setKinematics of three body interactions'
          collisionFlag=.false.
          finalState(1:2)%ID=0
          return
       end if
    end do
    finalState%scaleCS=1.
    finalState%inF=.false.

  end subroutine setKinematics


 !*****************************************************************************
 !****s* master_3Body/nukSearch
 ! PURPOSE
 ! Searches in particle Vector for those nucleons which are closest to the given particleIn. It
 ! returns pointers to the closest neutrons and protons.
 !*****************************************************************************
  subroutine NukSearch(particleIn,particleVector,proton1,proton2,neutron1,neutron2,flagOK)
    use densityModule
    use dichteDefinition
    use particleDefinition
    use random, only: rn
    use constants, only: rhoNull
    use collisionNumbering, only: check_justCollided
    use IDTable, only: nucleon
    use inputGeneral, only: continousboundaries, fullEnsemble, numensembles

    type(particle), intent(in) :: particleIn                              ! Regarded single particle
    type(particle), dimension(:,:), intent(in), target :: particleVector  ! field of possible scattering partners
    type(particle),pointer :: proton1,proton2,neutron1,neutron2           ! Closest protons & neutrons
    logical, intent(out) :: flagOK

    real, dimension(1:3) :: position, ortsAbstand
    integer, dimension(0:1) :: nucleonsFound       ! Number of found nucleons, 0=neutrons, 1=protons
    type(dichte) :: dens
    real :: radius, distance, distance_crossingEdge
    integer :: i,j,iStart,jStart, i_index,j_index,jj

    if (initFlag) call readInput

    position=particleIn%pos

    ! Search for nucleons
    nucleonsFound(0:1)=0          ! count nucleons that were found

    ! Initialize indices
    flagOK=.true.

    ! Initialize pointers
    proton1  => null()
    proton2  => null()
    neutron1 => null()
    neutron2 => null()

    dens=densityAt(position)

    if (dens%baryon(0).lt.5e-03) then          !If density too small, then no absorption
!       If(debug) write(*,*) 'density too small:', dens%baryon,'#####',position
       flagOK=.false.
       return
    else
 !      If(debug) write(*,*) 'density ok:', dens%baryon,'#####',position
       radius=Min(radiusNukSearch*(rhoNull/dens%baryon(0))**(1./3.),5.0)

       if (fullensemble) radius=radius/(float(numensembles)**(1./3.))

       !randomize starting point for loop over particles
       iStart=int(rn()*size(particleVector,dim=1))
       jStart=int(rn()*size(particleVector,dim=2))

       dim1_Loop: do j=1,size(particleVector,dim=2)
          j_index=mod(j+jstart,size(particleVector,dim=2))+lbound(particleVector,dim=2)
          dim2_loop : do i=1,size(particleVector,dim=1)
             i_index=mod(i+istart,size(particleVector,dim=1))+lbound(particleVector,dim=1)
             if (particleVector(i_index,j_index)%ID.ne.nucleon &
               & .or. particleVector(i_index,j_index)%anti) cycle dim2_loop

             ! Exclude nucleons which just interacted with the particle before
             if (check_justCollided(particleIn,particleVector(i_index,j_index)))cycle dim2_loop

             ! the direct distance
             distance=SQRT(Dot_Product(particleVector(i_index,j_index)%pos-position,&
                  & particleVector(i_index,j_index)%pos-position))

             if (continousboundaries) then
                ortsabstand=particleVector(i_index,j_index)%pos-position
                do jj=1,3
                   if (abs(ortsabstand(jj)).gt.gridsize(jj)) then
                      ortsabstand(jj)=2*gridsize(jj)-abs(ortsabstand(jj))
                   end if
                end do
                distance_crossingEdge=SQRT(Dot_Product(ortsabstand,ortsabstand))
             else
                distance_crossingEdge=10.*radius
             end if

             if (distance.le.radius.or.distance_crossingEdge.le.radius) then
                if ((particleVector(i_index,j_index)%charge.eq.0).and.(nucleonsFound(0).le.1)) then ! neutron found
                   nucleonsFound(0)=nucleonsFound(0)+1
                   if (nucleonsFound(0).eq.1) then
                      neutron1=>particleVector(i_index,j_index)
                   else
                      neutron2=>particleVector(i_index,j_index)
                   end if

                else if ((particleVector(i_index,j_index)%charge.eq.1).and.(nucleonsFound(1).le.1)) then ! proton found
                   nucleonsFound(1)=nucleonsFound(1)+1
                   if (nucleonsFound(1).eq.1) then
                      proton1=>particleVector(i_index,j_index)
                   else
                      proton2=>particleVector(i_index,j_index)
                   end if
                end if
             end if
             if (Sum(nucleonsFound).eq.4) exit dim1_Loop!enough nucleons are found
          end do dim2_loop
       end do dim1_Loop
       if ((nucleonsFound(1)+nucleonsFound(0)).ne.4) flagOK=.false. !four nucleons could not be found
    end if

!    If (debug) then
!       write(*,*) 'nach Nuksuche',nucleonsFound
!    end if

    if (debug) then
       open(973,File="NukSuche_ThreeBody.dat",position="Append")
       if (nucleonsFound(1)+nucleonsFound(0).lt.2) then
          write(973,'(A,2I3,5F9.3)') 'nucleonsFound <2',nucleonsFound(1),nucleonsFound(0),radius,position,dens%baryon(0)
       else if (nucleonsFound(1)+nucleonsFound(0).eq.2) then
          write(973,'(A,2I3,5F9.3)') 'nucleonsFound =2',nucleonsFound(1),nucleonsFound(0),radius,position,dens%baryon(0)
       else if (nucleonsFound(1)+nucleonsFound(0).eq.3) then
          !three nucleons have been found
          write(973,'(A,2I3,5F9.3)')'nucleonsFound =3',nucleonsFound(1),nucleonsFound(0),radius,position,dens%baryon(0)
       else if (nucleonsFound(1)+nucleonsFound(0).eq.4) then
          write(973,'(A,2I3,5F9.3)')'nucleonsFound =4',nucleonsFound(1),nucleonsFound(0),radius,position,dens%baryon(0)
       end if
       close(973)
    end if
  end subroutine NukSearch

end module master_3Body
