!******************************************************************************
!****m* /collisionTerm
! NAME
! module collisionTerm
! PURPOSE
! Includes the collision term.
!******************************************************************************
module collisionTerm

  use CallStack, only: Traceback

  implicit none
  private

  !****************************************************************************
  !****g* collisionTerm/useStatistics
  ! SOURCE
  !
  logical, save :: useStatistics=.false.
  ! PURPOSE
  ! Generate statistical information using the module statistics.
  !****************************************************************************

  !****************************************************************************
  !****g* collisionTerm/noNucNuc
  ! SOURCE
  !
  logical, save :: noNucNuc=.false.
  ! PURPOSE
  ! Switch on/off perturbative NN reactions.
  !****************************************************************************


  !****************************************************************************
  !****g* collisionTerm/energyCheck
  ! SOURCE
  !
  real, save :: energyCheck=0.01
  ! PURPOSE
  ! Precision of energy check for each collision in GeV.
  !****************************************************************************


  !****************************************************************************
  !****g* collisionTerm/oneBodyProcesses
  ! SOURCE
  !
  logical,save :: oneBodyProcesses=.true.
  ! PURPOSE
  ! Switch on/off one-body-induced processes.
  !****************************************************************************


  !****************************************************************************
  !****g* collisionTerm/oneBodyAdditional
  ! SOURCE
  !
  logical,save :: oneBodyAdditional=.true.
  ! PURPOSE
  ! Switch on/off additional Pythia one-body-induced processes.
  !****************************************************************************

  !****************************************************************************
  !****g* collisionTerm/twoBodyProcesses
  ! SOURCE
  !
  logical,save :: twoBodyProcesses=.true.
  ! PURPOSE
  ! Switch on/off two-body-induced processes.
  !****************************************************************************

  !****************************************************************************
  !****g* collisionTerm/twoBodyProcessesRealReal
  ! SOURCE
  !
  logical,save :: twoBodyProcessesRealReal=.true.
  ! PURPOSE
  ! Switch on/off two-body-induced processes between two real particles.
  !****************************************************************************

  !****************************************************************************
  !****g* collisionTerm/twoBodyProcessesRealPert
  ! SOURCE
  !
  logical,save :: twoBodyProcessesRealPert=.true.
  ! PURPOSE
  ! Switch on/off two-body-induced processes between a
  ! real and a perturbative particle.
  !****************************************************************************

  !****************************************************************************
  !****g* collisionTerm/threeBodyProcesses
  ! SOURCE
  !
  logical,save :: threeBodyProcesses=.true.
  ! PURPOSE
  ! Switch on/off three-body-induced processes.
  !****************************************************************************

  !****************************************************************************
  !****g* collisionTerm/threeMesonProcesses
  ! SOURCE
  !
  logical,save :: threeMesonProcesses=.false.
  ! PURPOSE
  ! Switch on/off three-meson-induced processes.
  ! These are the backreactions for e.g. omega -> pi pi pi etc.
  !****************************************************************************

  !****************************************************************************
  !****g* collisionTerm/threeBarMesProcesses
  ! SOURCE
  !
  logical,save :: threeBarMesProcesses=.false.
  ! PURPOSE
  ! Switch on/off baryon-meson-meson induced processes.
  ! These are the backreactions for e.g. N pi -> N pi pi etc.
  !****************************************************************************

  !****************************************************************************
  !****g* collisionTerm/twoPlusOneBodyProcesses
  ! SOURCE
  !
  logical,save :: twoPlusOneBodyProcesses=.false.
  ! PURPOSE
  ! Switch on/off 2+1 body processes
  ! (two really colliding particles plus one nearby).
  !****************************************************************************

  !****************************************************************************
  !****g* collisionTerm/doForceDecay
  ! SOURCE
  !
  logical,save :: doForceDecay = .true.
  ! PURPOSE
  ! switch on/off the forced decays at the end of the simulation
  ! NOTES
  ! * Do not touch, unless you know what you are doing!
  ! * You may set this to .false., if you are e.g. running box calculations
  !   with excited states. Decaying all these particles would need a much
  !   larger particle vector...
  !****************************************************************************

  !****************************************************************************
  !****g* collisionTerm/maxOut
  ! SOURCE
  !
  integer,save :: maxOut=100
  ! PURPOSE
  ! Maximal number of produced particles in one process.
  ! Must not be smaller than 4.
  !
  ! NOTES
  ! * When using annihilation, must not be smaller than 6
  ! * If using FRITIOF or PYTHIA, should probably larger than 6
  ! * Code stops with an error message, if value chosen too small
  !****************************************************************************

  !****************************************************************************
  !****g* collisionTerm/printPositions
  ! SOURCE
  !
  logical,save :: printPositions=.false.
  ! PURPOSE
  ! Switch on/off output of positions in real-pert collisions.
  ! Produces statistical output.
  !****************************************************************************

  !****************************************************************************
  !****g* collisionTerm/storeRho0Info
  ! SOURCE
  !
  logical, save :: storeRho0Info = .false.
  ! PURPOSE
  ! Flag whether in a rho0 decay the particle numbers of the
  ! resulting charged pions are stored or not.
  !****************************************************************************

  !****************************************************************************
  !****g* collisionTerm/storeRho0InfoOnlyDifr
  ! SOURCE
  !
  logical, save :: storeRho0InfoOnlyDifr = .false.
  ! PURPOSE
  ! Flag, whether the flag storeRho0Info is valid for all decays or
  ! only for rho0, which are marked to be diffractive.
  !****************************************************************************

  !****************************************************************************
  !****g* collisionTerm/DoJustAbsorptive
  ! SOURCE
  !
  logical, save :: DoJustAbsorptive = .false.
  ! PURPOSE
  ! If this flag is true, then:
  ! for perturbative simulations all final state particles in a collision are
  ! set to zero; for real simulations %event index of incoming hadron is changed
  ! in the case of collision,
  ! but actual collision is not simulated.
  ! This is a way to mimick Glauber like calculations.
  ! NOTES
  ! The "absorption" is done with sigmaTot, not just by sigmaInEl.
  !****************************************************************************


  !****************************************************************************
  !****g* collisionTerm/annihilate
  ! SOURCE
  !
  logical, save :: annihilate = .false.
  ! PURPOSE
  ! If this flag is true, then an annihilation of the antibaryons with
  ! the closest baryons will be simulated (by hand) starting from
  ! annihilationTime.
  !****************************************************************************


  !****************************************************************************
  !****g* collisionTerm/annihilationTime
  ! SOURCE
  !
  real, save :: annihilationTime = 1000.
  ! PURPOSE
  ! Time moment (in fm/c) when the annihilation will be started.
  ! NOTES
  ! This flag has an influence only when annihilate = .true.
  ! Before annihilationTime all the collision processes are not
  ! activated. They start to act (if the corresponding switches
  ! oneBodyProcesses,twoBodyProcesses etc. are .true.) only after
  ! annihilationTime.
  !****************************************************************************

  !****************************************************************************
  !****g* collisionTerm/justDeleteDelta
  ! SOURCE
  !
  logical, save :: justDeleteDelta = .false.
  ! PURPOSE
  ! Deletes final-state products in Delta N N -> NNN and Delta N -> N N.
  ! NOTE: Only for testing and comparing with the old Effenberger code!
  ! DO NOT USE OTHERWISE: Violates energy conservation!
  ! This switch is meant to simulate the treatment of the Delta in the old code.
  ! Only implemented for perturbative runs.
  !****************************************************************************

  !****************************************************************************
  !****g* collisionTerm/collisionProtocol
  ! SOURCE
  !
  logical, save :: collisionProtocol = .false.
  ! PURPOSE
  ! Write a protocol of all real-real collisions to the file 'fort.990'.
  ! Includes the time, IDs, charges, invariant masses and 3-momenta of both
  ! collision partners and an indicator for Pauli blocking.
  !****************************************************************************

  !****************************************************************************
  !****g* collisionTerm/noRecollisions
  ! SOURCE
  !
  logical, save :: noRecollisions = .false.
  ! PURPOSE
  ! Outgoing particles of collisions are inserted somewhere in the particle
  ! vector. Due to implementation issues, these outgoing particles may interact
  ! during the same timestep.
  !
  ! Setting this flag to true, the parameter '%lastCollTime' is checked
  ! against the actual time variable and collisions of these particles are
  ! excluded.
  !****************************************************************************

  !****************************************************************************
  !****g* collisionTerm/debug
  ! SOURCE
  logical, parameter :: debug=.false.
  ! PURPOSE
  ! Switch for debug information.
  !****************************************************************************

  logical, parameter :: debugSaveInfo=.false.
  logical, save :: initFlag=.true.

!!$  type(histogram), save :: hXStot,hXSelast

  public :: collideMain, ForceDecays
  public :: readInput

contains


  !****************************************************************************
  !****s* collisionTerm/readInput
  ! NAME
  ! subroutine readInput
  ! PURPOSE
  ! Reads input in jobcard out of namelist "collisionTerm".
  !****************************************************************************
  subroutine readInput

    use inputGeneral, only: freezeRealParticles
    use output, only: Write_ReadingInput !,notInRelease
    use collisionTools, only: setEnergyCheck

    integer :: ios

    !**************************************************************************
    !****n* collisionTerm/collisionterm
    ! PURPOSE
    ! Namelist for collisionTerm includes:
    ! * oneBodyProcesses
    ! * twoBodyProcesses
    ! * threeBodyProcesses
    ! * threeMesonProcesses
    ! * threeBarMesProcesses
    ! * twoPlusOneBodyProcesses
    ! * twoBodyProcessesRealReal
    ! * twoBodyProcessesRealPert
    ! * oneBodyAdditional
    ! * doForceDecay
    ! * energyCheck
    ! * maxOut
    ! * collisionProtocol
    ! * printPositions
    ! * useStatistics
    ! * noNucNuc
    ! * storeRho0Info
    ! * storeRho0InfoOnlyDifr
    ! * DoJustAbsorptive
    ! * annihilate
    ! * annihilationTime
    ! * justDeleteDelta
    ! * noRecollisions
    !**************************************************************************
    NAMELIST /collisionTerm/ &
         oneBodyProcesses, twoBodyProcesses, threeBodyProcesses, &
         twoPlusOneBodyProcesses, &
         threeMesonProcesses, threeBarMesProcesses, &
         twoBodyProcessesRealReal, twoBodyProcessesRealPert, &
         oneBodyAdditional, doForceDecay, &
         energyCheck, maxOut, collisionProtocol, printPositions, useStatistics,&
         noNucNuc,storeRho0Info, storeRho0InfoOnlyDifr, DoJustAbsorptive, &
         annihilate, annihilationTime, justDeleteDelta, &
         noRecollisions

    rewind(5)
    call Write_ReadingInput('collisionTerm',0)
    read(5,nml=collisionTerm,iostat=ios)

    if (ios>0) then
       write(*,*)
       write(*,*) '>>>>> parameter "minimumEnergy" moved to the new &
                   & namelist "insertion"'
    end if
    call Write_ReadingInput('collisionTerm',0,ios)
    write(*,*) 'Precision of energy Check in GeV=',energyCheck

    write(*,*) 'Do 1-body-induced reactions :', oneBodyProcesses, &
         '  ...additional Jetset: ',oneBodyAdditional

    ! override input if overall flag forbids collisions
    if (freezeRealParticles) twoBodyProcessesRealReal = .false.

    if (twoBodyProcesses) then
       write(*,*) 'Do 2-body-induced reactions :', twoBodyProcesses, &
            '  ...RealReal,RealPert=',twoBodyProcessesRealReal,      &
                  & twoBodyProcessesRealPert
    else
       write(*,*) 'Do 2-body-induced reactions :', twoBodyProcesses
    end if
    write(*,*) 'Do 3-body-induced reactions :', threeBodyProcesses
    write(*,*) 'Do 2+1 body collisions      :', twoPlusOneBodyProcesses
    write(*,*) 'Do 3-meson reactions        :', threeMesonProcesses
    write(*,*) 'Do bar-mes-mes reactions    :', threeBarMesProcesses
    write(*,*) 'Forbid recollisions         :', noRecollisions
    write(*,*) 'Force decays at end         :', doForceDecay
    write(*,*)

    write(*,*) 'Maximal number of final state particles :', maxOut
    if (maxOut<4) &
         call Traceback("maxout must not be smaller than 4!")

    write(*,*) 'protocol collisions?', collisionProtocol
    write(*,*) 'print positions    ?', printPositions
    write(*,*) 'Statistical output ?' ,useStatistics
    write(*,*) 'Switching off perturbative NN collisions:',noNucNuc
    write(*,*) 'Store pion numbers in rho0 decay:        ',storeRho0Info
    write(*,*) '              -"-      only diffractive: ',storeRho0InfoOnlyDifr

    if (DoJustAbsorptive) then
       write(*,*)
       write(*,*) 'ATTENTION: absorptive treatment !!!!'
       write(*,*)
    end if


    if (annihilate) then
       write(*,*)
       write(*,*) 'ATTENTION: annihilation of the antibaryons by hand !!!!'
       write(*,*) '  annihilation time:  ', annihilationTime
       write(*,*)
    end if

    if (justDeleteDelta) then
       write(*,*)
       write(*,*) 'ATTENTION: Delta N N -> NNN and Delta N -> NN:   &
                  & Finalstate not populated !!!!'
       write(*,*) 'ATTENTION: ONLY FOR TESTING,  NOT FOR REGULAR USE !!!!'
       write(*,*)
    end if

    if (threeMesonProcesses) then
!       call notInRelease("threeMesonProcesses")
    end if

    call Write_ReadingInput('collisionTerm',1)
    initFlag = .false.

!!$    call CreateHist(hXStot,   'XS tot',   0.,10.,0.05)
!!$    call CreateHist(hXSelast, 'XS elast', 0.,10.,0.05)

    call setEnergyCheck(energyCheck)

  end subroutine readInput


  !****************************************************************************
  !****s* collisionTerm/ForceDecays
  ! NAME
  ! subroutine ForceDecays(partPert, partReal, time)
  ! PURPOSE
  ! This routine forces all instable particles to decay in the last timestep.
  ! We loop over the particle vectors several times in order to allow for
  ! multi-step decay chains and to make sure that everything has decayed.
  ! INPUTS
  ! * type(particle), dimension(:,:) :: partReal -- real particles
  ! * type(particle), dimension(:,:) :: partPert -- perturbative particles
  ! * real                           :: time
  !****************************************************************************
  subroutine ForceDecays(partPert, partReal, time)
    use particleDefinition
    use output, only: DoPr
    use AddDecay, only: PerformAddDecay
    use Dilepton_Analysis, only: Dilep_Decays
    use inputGeneral, only: delta_t, eventType
    use eventTypes, only: HeavyIon, hadron

    type(particle), dimension(:,:) :: partReal, partPert
    real, intent(in) :: time
    integer :: i,l
    logical :: dec

    ! Initialize at first call
    if (initFlag) call ReadInput

    if (.not.doForceDecay) return

    !       call WriteHist(hXStot,121,file="CollTerm.XS.tot.dat",DoAve=.true.)
    !       call WriteHist(hXSElast,121,file="CollTerm.XS.elast.dat",DoAve=.true.)

    if (oneBodyProcesses) then
       ! At the end of all time steps the resonances are forced to decay:
       do i=1,100
          ! Evaluate final decays
          if (DoPr(2)) write(*,*) 'No usual Collision Term;   &
              &  Just particle decays: ', i
          dec = oneBody(partPert,partReal,time,.true.)

          ! final Dilepton Analysis during/after forced decays
          if (dec) then
             l = 1
          else
             l = 2
          end if
          if (eventType==HeavyIon .or. eventType==hadron) then
             call Dilep_Decays(time+i*delta_t, partReal, l)
          else
             call Dilep_Decays(time+i*delta_t, partPert, l)
          end if

          if (.not. dec) exit  ! no further decays have occurred => exit the loop
       end do

       if (oneBodyAdditional) then
          if (DoPr(2)) write(*,*) 'Perform additional Jetset Decays.'
          call PerformAddDecay(partReal,time)
          call PerformAddDecay(partPert,time)
       end if
    end if

  end subroutine ForceDecays


  !****************************************************************************
  !****s* collisionTerm/collideMain
  ! NAME
  ! subroutine collideMain(partPert, partReal, time)
  ! PURPOSE
  ! Evaluates the collision term for a given real and perturbative particle
  ! vector.
  ! There are reactions induced by a single particle (decays),
  ! two particles and even three particles.
  ! INPUTS
  ! * type(particle), dimension(:,:) :: partReal -- real particles
  ! * type(particle), dimension(:,:) :: partPert -- perturbative particles
  ! * real                           :: time
  !****************************************************************************
  subroutine collideMain(partPert, partReal, time)
    use particleDefinition
    use output, only: DoPr, paragraph
    use deuterium_Pl, only: deuterium_PertOrigin!,deuterium_pertOrigin_flag
    use inputGeneral, only: localEnsemble
    use CollHistory, only: CollHist_SetSize
    use AntibaryonWidth, only: DoAnnihilation
    use ThreeMeson, only: doThreeMeson
    use ThreeBarMes, only: doThreeBarMes


    type(particle), intent(inout), dimension(:,:) :: partPert, partReal
    real, intent(in) :: time

    logical :: dec

    ! Initialize at first call
    if (initFlag) call ReadInput

    if (size(partPert,dim=1) /= size(partReal,dim=1)) then
       write(*,*) 'Number of ensembles   &
                  & in real and perturbative particle vectors do not fit'
       write(*,*) 'Real :' , size(partReal,dim=1)
       write(*,*) 'Perturbative :' , size(partPert,dim=1)
       call Traceback('Critical Error in collideMain! Stop!')
    end if

    if (annihilate) call DoAnnihilation(partReal,time,annihilationTime)

    ! prepare for extra collision history
    call CollHist_SetSize(ubound(partPert))

    ! Evaluate decays
    if (oneBodyProcesses) then
       if (DoPr(2)) write(*,paragraph) "1-Body Processes"
       dec = oneBody(partPert, partReal, time, .false.)
    end if

    ! Evaluate two-body collisions
    if (twoBodyProcesses) then
       if (DoPr(2)) write(*,paragraph) "2-Body Processes"
       if (localEnsemble) then
          call twoBody_local(partPert, partReal, time)
       else
          call twoBody(partPert, partReal, time)
       end if
    end if

    ! Evaluate three-body interactions
    if (threeBodyProcesses) then
       if (DoPr(2)) write(*,paragraph) "3-Body Processes"
       call threeBody(partPert, partReal, time)
    end if

    ! Evaluate three-meson interactions
    if (threeMesonProcesses) then
       if (DoPr(2)) write(*,paragraph) "3-Meson Processes"
       call DoThreeMeson(time)
    end if

    ! Evaluate BarMesMes interactions
    if (threeBarMesProcesses) then
       if (DoPr(2)) write(*,paragraph) "3-Baryon-Meson Processes"
       call DoThreeBarMes(time)
    end if

    nullify(deuterium_pertOrigin)

  end subroutine collideMain



  !****************************************************************************
  !****s* collisionTerm/oneBody
  ! NAME
  ! logical function oneBody(partPert, partReal, time, ForceFlag)
  ! PURPOSE
  ! Administrates the 1-body processes (i.e. decays).
  !
  ! INPUTS
  ! * type(particle),dimension(:,:) :: partPert -- perturbative particles
  ! * type(particle),dimension(:,:) :: partReal -- real particles
  ! * real    :: time      -- actual time step
  ! * logical :: ForceFlag --
  !   .true. = let particles decay (decay probability=1),
  !   do not consider the width anymore for the decay probability.
  !   This is useful at the end of a run.
  !
  ! OUTPUT
  ! * partPert and partReal are changed.
  ! * The return value indicates whether any decays have occurred.
  !****************************************************************************
  logical function oneBody(partPert, partReal, time, ForceFlag)

    use master_1Body, only: decayParticle
    use inputGeneral, only: fullEnsemble, numEnsembles, freezeRealParticles
    use collisionNumbering, only: ReportEventNumber
    use particleDefinition
    use output, only: WriteParticleVector
    use IdTable, only: isMeson
    use deuterium_PL, only: deuterium_pertOrigin,deuterium_pertOrigin_flag
    use pauliBlockingModule, only: checkPauli
    use twoBodyStatistics, only: rate
    use insertion, only: particlePropagated, setIntoVector
    use collisionTools, only: finalCheck

    type(particle), intent(inout), dimension(:,:) :: partPert, partReal
    real, intent(in) :: time
    logical, intent(in) :: ForceFlag

    integer :: index,ensemble
    type(particle),target         :: resonance   ! particle which shall decay
    type(particle),dimension(1:3) :: finalState  ! final state of particles
    logical :: Flag, setFlag

    oneBody = .false.

    if (debug) write(*,*) 'real decays'
    if (.not.freezeRealParticles) call RealDecay
    if (debug) write(*,*) 'perturbative decays'
    call PertDecay
    if (debug) write(*,*) 'after perturbative decays'

  contains

    !**************************************************************************
    ! real particles decaying
    !**************************************************************************
    subroutine RealDecay
      use idTable, only: photon

      integer :: i

      ensemble_loop : do ensemble=1,numEnsembles
         index_loop : do index=1,size(partReal,dim=2)

            if (partReal(ensemble,index)%ID < 0) exit index_loop !!! ID==-1
            if (partReal(ensemble,index)%ID == 0) cycle index_loop
            if (partReal(ensemble,index)%ID == photon) cycle index_loop

            resonance=partReal(ensemble,index)
            deuterium_pertOrigin=>resonance
            deuterium_pertOrigin_flag=11

            call decayParticle(resonance,finalState,flag,ForceFlag, &
                              & time)
            if (.not.ForceFlag) then
               ! Note : If particles are forced to decay then we forget
               !        about Pauli blocking
               if (.not. flag) cycle index_loop
               flag = checkPauli(finalState,partReal)
            end if
            if (.not.flag) cycle index_loop

            flag = finalCheck((/resonance/), finalState)
            if (.not.flag) cycle index_loop

            oneBody = .true. ! at this point we know that the decay is happening

            ! Compute various collision rates:
            call rate((/resonance/),finalState,time)


            ! (1) Eliminate incoming particle
            partReal(ensemble,index)%ID = 0

            ! (2) Create final state particles:
            do i=1,3
               if (finalState(i)%id <= 0) exit
               if (particlePropagated(finalState(i))) then
                  if (partReal(ensemble,index)%ID.eq.0) then

                     call setNumber(finalState(i))
                     partReal(ensemble,index)=finalState(i)

                  else
                     ! Find empty space in the particle vector:
                     if (fullensemble) then
                        call setIntoVector(finalState(i:i),partReal,setFlag)
                     else
                        call setIntoVector(finalState(i:i),   &
                                      &partReal(ensemble:ensemble,:),setFlag)
                     end if
                     ! (3) Check that setting into real particle
                     !     vector worked out :
                     if (.not. setFlag) then
                        write(*,*) 'Real particle vector too small!'
                        write(*,*) size(finalState),&
                             lbound(partReal), ubound(partReal)
                        write(*,*) 'Dumping real particles to files &
                                   & "RealParticles_stop_*!"'
                        call WriteParticleVector("RealParticles_stop",partReal)
                        write(*,*) 'Dumping perturbative particles to files &
                                   &"PertParticles_stop_*!"'
                        call WriteParticleVector("PertParticles_stop",partPert)
                        call Traceback('collision Term, one-body')
                     end if
                  end if
               end if
            end do

            call ReportEventNumber((/resonance/),finalState,  &
                 finalState(1)%event,time,11, &
                 iEns1=ensemble)

         end do index_loop ! Index of particle
      end do ensemble_loop ! Ensemble of particle
      nullify(deuterium_pertOrigin)

    end subroutine RealDecay

    !**************************************************************************
    ! perturbative decays
    !**************************************************************************
    subroutine PertDecay
      use idTable, only: photon, pion, rho, nucleon
      use statistics, only: saveInfo
      use PIL_rho0Dec, only: PIL_rho0Dec_PUT
      use CollHistory, only: CollHist_UpdateHist

      integer :: k
      logical :: r!,rr
      integer, dimension(2,3) :: posOut

      !loop over first particle=real particle:
      ensemble_loop : do ensemble=1,numEnsembles
         index_loop : do index=1,size(partPert,dim=2)

            if (partPert(ensemble,index)%ID < 0) exit index_loop !!! ID==-1
            if (partPert(ensemble,index)%ID == 0) cycle index_loop
            if (partPert(ensemble,index)%ID.eq.photon) cycle index_loop

            resonance=partPert(ensemble,index)
            deuterium_pertOrigin=>resonance
            deuterium_pertOrigin_flag=1

            call decayParticle(resonance,finalState,flag,ForceFlag,time)
            if (.not.ForceFlag) then
               ! Note : If particles are forced to decay then we forget
               !        about Pauli blocking
               if (.not.flag) cycle index_loop
               flag = checkPauli(finalState,partReal)
            end if
            if (.not.flag) cycle index_loop

            flag = finalCheck((/resonance/), finalState)
            if (.not.flag) cycle index_loop

            oneBody = .true. ! at this point we know that the decay is happening

            ! Compute various collision rates:
            call rate((/resonance/),finalState,time)

            ! (1a) some statistics

            do k=lbound(finalstate,dim=1),ubound(finalstate,dim=1)
               if (finalState(k)%ID.le.0) cycle
               call setNumber(finalState(k))
               ! Statistical output
               if (printPositions .and. (finalState(k)%id==pion .or. &
                   & finalState(k)%id==nucleon)) &
                    call positionPrinter(finalstate(k))
               if (useStatistics) then
                  call saveInfo(finalState(k)%number,ensemble,  &
                               & finalstate(k)%pos, resonance%ID,0,0)
                  if (debugSaveInfo) write(98,*)finalState(k)%number,ensemble, &
                     & finalstate(k)%pos, resonance%ID
               end if
            end do

            ! (1b) Store information for a rho0 Decay
            if (storeRho0Info .and. resonance%ID==rho .and. resonance%charge==0)&
               & then
               r = .true.
               !if (storeRho0InfoOnlyDifr)  &
               !   & rr = PIL_rhoDiffractive_GET(resonance(1)%number,r)

               if (r) then
                  call PIL_rho0Dec_PUT(finalState(1)%number,   &
                                      & finalState(2)%number, &
                     & resonance%lastCollTime, resonance%prodTime, &
                     &  resonance%formTime)
                  call PIL_rho0Dec_PUT(finalState(2)%number, &
                                      &  finalState(1)%number, &
                      &  resonance%lastCollTime, resonance%prodTime,&
                      &  resonance%formTime)
               end if
            end if

            ! (2) Eliminate incoming particle
            partPert(ensemble,index)%Id = 0
            ! (3) Create final state particles
            posOut = 0
            if (particlePropagated(finalState(1))) then
               partPert(ensemble,index) = finalState(1)
               posOut(1:2,1) = (/ensemble,index/)
            end if
            if (fullensemble) then
               call setIntoVector(finalState(2:), partPert, setFlag, .true., &
                                 & positions=posOut(:,2:))
            else
               call setIntoVector(finalState(2:), partPert(ensemble:ensemble,:),&
                                 & setFlag, .true., positions=posOut(:,2:))
               posOut(1,2:) = ensemble
            end if

            call ReportEventNumber((/resonance/), finalState, &
                 finalState(1)%event, time, 12, &
                 iEns1=ensemble)

            call CollHist_UpdateHist((/resonance/), finalState,  &
                             & (/ensemble,index/), posOut, resonance%perweight)

            ! (4) Check that setting into perturbative particle vector worked out
            if (.not.setFlag) then
               write(*,*) 'Perturbative particle vector too small!'
               write(*,*) size(finalState),  lbound(partPert), ubound(partPert)
               write(*,*) 'Dumping real particles to files "realParticles_stop_*!"'
               call WriteParticleVector("RealParticles_stop",partReal)
               write(*,*) 'Dumping perturbative particles to files &
                          &"PertParticles_stop_*!"'
               call WriteParticleVector("PertParticles_stop",partPert)
               call Traceback('collision Term, one-body')
            end if

            if (debug) write(*,*) 'After Set into vector'

         end do index_loop ! Index of particle
      end do ensemble_loop! Ensemble of  particle
      nullify(deuterium_pertOrigin)

    end subroutine PertDecay

  end function oneBody


  !****************************************************************************
  !****s* collisionTerm/twoBody
  ! NAME
  ! subroutine twoBody(partPert, partReal, time)
  ! PURPOSE
  ! Administrates the 2-body processes.
  ! The collision criteria is based upon the Kodama criteria.
  !
  ! INPUTS
  ! * type(particle),dimension(:,:) :: partPert -- perturbative particles
  ! * type(particle),dimension(:,:) :: partReal -- real particles
  ! * real                          :: time         -- actual time step
  !
  ! OUTPUT
  ! * partPert and partReal are changed
  !****************************************************************************
  subroutine twoBody(partPert, partReal, time)

    use master_2Body, only: collide_2body
    use twoBodyStatistics, only: sqrts_distribution, rate
    use inputGeneral, only: fullEnsemble,eventtype,numEnsembles
    use eventtypes, only: HeavyIon
    use particleDefinition
    use output, only: WriteParticleVector, timeMeasurement
    use IdTable, only: isMeson, isBaryon
    use master_NBody, only: NBody_Analysis, check_for_Nbody, &
        & generate_3body_collision
    use XsectionRatios, only: accept_event
    use deuterium_PL, only: deuterium_pertOrigin,deuterium_pertOrigin_flag,  &
        & deuteriumPL_ensemble
    use pauliBlockingModule, only: checkPauli
    use initHadron, only: particleId,antiparticle,particleCharge
    use insertion, only: particlePropagated, setIntoVector
    use collisionTools, only: finalCheck
    use residue, only: ResidueAddPH

    type(particle), intent(inout), target, dimension(:,:) :: partPert, partReal
    real, intent(in) :: time

    integer :: index1,index2,ensemble1,ensemble2,index1_start,index2_end,  &
               & ensemble2_start,ensemble2_end  ! Loop variables
    type(particle), dimension(1:maxOut) :: finalState
    ! final state of particles in 2-body collisions
    type(particle), dimension(1:maxOut) :: finalState_3Body
    ! final state of particles in 3-body collisions
    logical :: Flag, Flag_3Body, setFlag, HiEnergyFlag, HiEnergyFlag_3Body, &
               flag1, flag2, flag3
    integer :: size_pert_dim2, size_real_dim2
    integer :: HiEnergyType, HiEnergyType_3Body
    integer :: i,number
    integer :: justdeletedelta_count=0,justdeletedelta_countO=0

    size_real_dim2=size(partReal,dim=2)
    size_pert_dim2=size(partPert,dim=2)

    if (twoBodyProcessesRealReal) call realReal
    if (twoBodyProcessesRealPert) call realPert

  contains

    !**************************************************************************
    ! real<->real collisions
    !**************************************************************************
    subroutine realReal

      use idTable, only: photon
      use collisionNumbering, only: check_justCollided, real_numbering,   &
          & real_firstnumbering, ReportEventNumber
      use minkowski, only: abs4
      use twoBodyTools, only: isElastic

      type(particle), pointer :: part1, part2, part3
      type(particle), dimension(1:2) :: partIn

      !************************************************************************
      ! Variables needed for many-body collisions:
      integer :: n_found ! number of particles found around colliding pair
      integer, parameter :: Nmax=400 ! max possible value of n_found
      integer :: ind_min ! index of the most close particle to the collision point
      integer, dimension(1:Nmax) :: ind_found ! indices of found particles
      real :: coll_time  ! collision time instant
      real :: sigma,gamma ! cross section (mbarn) and gamma-factor in CM frame &
      ! of colliding particles
      !************************************************************************
      logical :: numbersAlreadySet,pauliIsIncluded
      integer :: imax,isuccess

      deuterium_pertOrigin_flag=99
      nullify(deuterium_pertOrigin)

      if (debug) write(*,*) 'real<->real'
      if (debug) call timeMeasurement(.true.)


      ensemble1_loop : do ensemble1=1,numEnsembles

         if (fullEnsemble) then
            ensemble2_End=numEnsembles
         else
            ensemble2_End=ensemble1
         end if

         ! loop over second ensemble:
         ensemble2_loop : do ensemble2=ensemble1,ensemble2_End

            if (ensemble2==ensemble1) then
               index1_start=2
            else
               index1_start=1
            end if

            deuteriumPL_ensemble=ensemble2

            index1_loop : do index1=index1_start,size_real_dim2

               part1 => partReal(ensemble1,index1)
               if (part1%ID < 0) exit index1_loop !!! ID==-1
               if (part1%ID <= 0) cycle index1_loop
               if (part1%ID.eq.photon) cycle index1_loop

               if (noRecollisions.and.(part1%lastCollTime>time-1e-6))  &
                  & cycle index1_loop

               if (ensemble2.eq.ensemble1) then
                  index2_end=index1-1
               else
                  index2_end=size_real_dim2
               end if

               if (DoJustAbsorptive .and. part1%event(1).ge.1000000)   &
                  & cycle index1_loop

               index2_loop : do index2=1,index2_end

                  part2 => partReal(ensemble2,index2)
                  if (part2%ID < 0) exit index2_loop !!! ID==-1
                  if (part2%ID <= 0) cycle index2_loop
                  if (part2%ID.eq.photon) cycle index2_loop

                  if (noRecollisions.and.(part2%lastCollTime>time-1e-6)) &
                     & cycle index2_loop

                  if (DoJustAbsorptive .and. part2%event(1).ge.1000000)   &
                     & cycle index2_loop

                  ! Check that they didn't collide just before:
                  if (check_justCollided(part1,part2)) cycle index2_loop

                  call resetNumberGuess

                  call collide_2body((/part1,part2/), finalState, time, flag, &
                       HiEnergyFlag, HiEnergyType, &
                       collTime=coll_time, PauliIncluded_out=pauliIsIncluded)

                  coll_time=time+coll_time

                  if (.not.flag) cycle index2_loop

                  if (annihilate .and. HiEnergyType.eq.-3) cycle index2_loop

                  flag_3Body=.false.
                  part3=>partReal(ensemble1,1)
                  ! This is just dummy setting to avoid stop when
                  ! twoPlusOneBodyProcesses=.false.

                  ! Check for possible presence of another particles nearby
                  ! (currently only for heavy ion collisions, for the parallel
                  !  ensemble mode and only for the baryons):
                  if (twoPlusOneBodyProcesses .and. eventtype==HeavyIon .and. &
                     & .not.fullEnsemble &
                     & .and. (isBaryon(part1%ID) .or. isBaryon(part2%ID))) then

                     if (debug) write(*,*) 'before check_for_Nbody'

                     call check_for_Nbody((/index1,index2/),time,coll_time,&
                          & partReal(ensemble1,:),n_found,ind_found,ind_min, &
                          & sigma,gamma)

                     if (debug) write(*,*) 'after check_for_Nbody:', n_found

                     call Nbody_analysis((/index1,index2/),time,&
                          & partReal(ensemble1,:),n_found,ind_found,sigma,gamma)

                     if (ind_min.ne.0) then
                        ! Fill histograms for stat. analysis:
                        call sqrts_distribution((/part1,part2/),2)
                        part3=>partReal(ensemble1,ind_min)

                        call resetNumberGuess

                        call generate_3body_collision((/part1,part2,part3/),&
                             & finalState_3Body,flag_3Body,time,   &
                             & HiEnergyFlag_3Body,HiEnergyType_3Body)
                        if ( .not.flag_3Body ) cycle index2_loop
                     else
                        call sqrts_distribution((/part1,part2/),1)
                     end if

                  end if  ! End of many-body check

                  if (flag_3Body) then
                     finalState=finalState_3Body
                     HiEnergyFlag=HiEnergyFlag_3Body
                     HiEnergyType=HiEnergyType_3Body
                  end if

                  if (debug) write(*,*) 'flag_3Body:', flag_3Body

                  imax=0
                  do i=1,maxout
                     if (finalState(i)%id <= 0) exit
                     imax=i
                     if (debug) then
                        write(*,*) 'Id, antiparticle ?, charge, number:',&
                             &finalState(i)%Id,finalState(i)%anti,&
                             &finalState(i)%charge,finalState(i)%number
                     end if
                  end do

                  if (debug .and. flag_3Body) then
                     write(*,*) ' 3-d particle:'
                     write(*,*) part3%Id, part3%anti, part3%charge, &
                                & part3%mass
                     write(*,*) part3%mom
                     write(*,*) part3%vel
                     write(*,*) ' last final particle:'
                     write(*,*) finalState(imax)%Id, &
                                & finalState(imax)%anti, &
                                & finalState(imax)%charge, finalState(imax)%mass
                     write(*,*) finalState(imax)%mom
                     write(*,*) finalState(imax)%vel
                  end if

                  if (.not. flag_3Body) then
                     ! Accept or reject the event randomly,
                     ! using the in-medium cross section ratios:
                     if (.not.accept_event((/part1,part2/),finalState,HiEnergyFlag))  &
                        & cycle index2_loop
                  end if

                  if (.not.pauliIsIncluded) flag = checkPauli(finalState,partReal)

                  if (collisionProtocol) then
                     if (flag) then
                        isuccess = 1
                     else
                        isuccess = 0
                     end if
                     write(990,'(f6.1,2i4,4f8.4,2i4,4f8.4,i2)') time, &
                          & part1%ID, part1%charge, abs4(part1%mom), &
                          & part1%mom(1:3), &
                          & part2%ID, part2%charge, abs4(part2%mom), &
                          & part2%mom(1:3), isuccess
                  end if

                  if (.not. flag) cycle index2_loop

                  if (flag_3Body) then
                     if (debug) write(*,*) ' In realreal, HiEnergyFlag=', &
                                           & HiEnergyFlag
                     flag = finalCheck((/part1,part2,part3/), finalState, &
                                      & HiEnergyFlag)
                  else
                     flag = finalCheck((/part1,part2/), finalState, HiEnergyFlag)
                  end if

                  if (.not. flag) cycle index2_loop

                  if (finalState(3)%ID==0 .and. isElastic((/part1,part2/),  &
                     & finalState(1:2))) then
                     ! elastic
                     call sqrts_distribution((/part1,part2/),4)
                  else
                     ! inelastic
                     call sqrts_distribution((/part1,part2/),5)
                  end if

                  call rate((/part1,part2/),finalState,time)
                  ! Compute various collision rates

                  ! (1) Label event by eventNumber
                  number=real_numbering()
                  if (DoJustAbsorptive) then
                     if (part1%id==particleId .and.  &
                        & (part1%anti.eqv.antiparticle) .and. &
                        & part1%charge==particleCharge) then
                        part1%event=number
                        cycle index1_loop
                     else if (part2%id==particleId .and.   &
                             & (part2%anti.eqv.antiparticle) .and. &
                             & part2%charge==particleCharge) then
                        part2%event=number
                        cycle index2_loop
                     end if
                  end if
                  finalState%event(1)=number
                  finalState%event(2)=number

                  finalState%perweight = min(part1%perweight, part2%perweight)
                  number = min(part1%firstEvent, part2%firstEvent)
                  if (number==0) number = real_firstnumbering()
                  finalState%firstEvent = number
                  finalState%pert = .false.

                  ! (2) Create final state particles
                  NumbersAlreadySet = AcceptGuessedNumbers()

                  partIn = (/part1,part2/) ! do a Backup of the incoming parts

                  flag1=.false.
                  flag2=.false.
                  flag3=.false.
                  do i=1,imax
                     if (particlePropagated(finalState(i))) then
                        if (.not.flag1 .and. ((isBaryon(part1%ID) .and. &
                           & isBaryon(finalState(i)%ID) .and. &
                           & (part1%anti.eqv.finalState(i)%anti)) &
                           & .or. &
                           & (isMeson(part1%ID).and.isMeson(finalState(i)%ID))))&
                           & then
                           if (.not.NumbersAlreadySet .or.  &
                              &  finalState(i)%number==0)   &
                              &  call setNumber(finalState(i))
                           part1 = finalState(i)
                           flag1 = .true.
                        else if (.not.flag2 .and. ((isBaryon(part2%ID) .and. &
                                & isBaryon(finalState(i)%ID) .and. &
                                & (part2%anti.eqv.  &
                                & finalState(i)%anti)) .or. &
                             (isMeson(part2%ID) .and. &
                             & isMeson(finalState(i)%ID)))) then
                           if (.not.NumbersAlreadySet .or. &
                              & finalState(i)%number==0)   &
                              & call setNumber(finalState(i))
                           part2 = finalState(i)
                           flag2 = .true.
                        else if (flag_3Body .and. .not.flag3 .and. &
                             ((isBaryon(part3%ID) .and. isBaryon(finalState(i)%ID)&
                             & .and. &
                             (part3%anti.eqv.finalState(i)%anti))&
                             & .or. &
                             & (isMeson(part3%ID) .and. &
                             & isMeson(finalState(i)%ID)))) then
                           if (.not.NumbersAlreadySet .or. &
                              & finalState(i)%number==0)   &
                              & call setNumber(finalState(i))
                           part3 = finalState(i)
                           flag3 = .true.
                        else
                           ! Find empty space in the particle vector:
                           if (fullensemble) then
                              call setIntoVector (finalState(i:i), partReal,  &
                                   & setFlag, NumbersAlreadySet)
                           else
                              call setIntoVector (finalState(i:i), &
                                   & partReal(ensemble1:ensemble1,:), setFlag, &
                                   & NumbersAlreadySet)
                           end if
                           ! (3) Check that setting into real particle
                           !     vector worked out :
                           if (.not.setFlag) then
                              write(*,*) 'Real particle vector too small!'
                              write(*,*) size(finalState),&
                                   lbound(partReal), ubound(partReal)
                              write(*,*) 'Dumping real particles to files &
                                         & "RealParticles_stop_*!"'
                              call WriteParticleVector("RealParticles_stop", &
                                                      & partReal)
                              write(*,*) 'Dumping perturbative particles  &
                                         & to files "PertParticles_stop_*!"'
                              call WriteParticleVector("PertParticles_stop", &
                                                      & partPert)
                              call Traceback('collision Term, two-body')
                           end if
                        end if
                     end if
                  end do

                  ! For statistical purposes and also for possible calculation
                  ! of the time step
                  ! also a 3-body collision is treated here as a 2-body collision:
                  call ReportEventNumber(partIn, finalState, &
                       finalState(1)%event, time, 211, HiEnergyType, &
                       iEns1=ensemble1, iEns2=ensemble2)

                  ! Add a hole in the target nucleus (analysis only):
                  if(partIn(1)%event(1).lt.1000000) then
                     call ResidueAddPH(number,partIn(1))
                  else if(partIn(2)%event(1).lt.1000000) then
                     call ResidueAddPH(number,partIn(2))
                  end if

                  ! (4) destroy incoming particles if there are
                  ! no corresponding final leading particles:
                  if (.not.flag2) part2%Id=0
                  if (flag_3Body .and. .not.flag3) part3%Id=0
                  if (.not.flag1) then
                     part1%Id=0
                     cycle index1_loop
                  end if

               end do index2_loop ! Index2 of second particle
            end do index1_loop ! Index1 of first particle
         end do ensemble2_loop ! Ensemble2 of second particle
      end do ensemble1_loop ! Ensemble1 of first particle


      if (twoPlusOneBodyProcesses .and. eventtype==HeavyIon .and.  &
         & .not.fullEnsemble) &
           call Nbody_analysis((/index1,index2/),time,partReal(ensemble1,:), &
                & n_found,ind_found,sigma,gamma,.true.)


      deuterium_pertOrigin_flag=0

    end subroutine realReal

    !**************************************************************************
    ! real<->pert collisions
    !**************************************************************************
    subroutine realPert
      use IdTable, only: photon, nucleon, Delta, pion
      use statistics, only: saveInfo
      use CollHistory, only: CollHist_UpdateHist
      use collisionNumbering, only: check_justCollided, pert_numbering, &
          & pert_firstnumbering, ReportEventNumber

      type(particle),pointer :: part1, part2
      logical :: numbersAlreadySet
      integer :: number2
      logical :: pauliIsIncluded
      integer, dimension(2,maxOut) :: posOut
      type(particle) :: part2S

      if (debug) write(*,*) 'real<->pert'
      if (debug) call timeMeasurement(.false.)
      if (debug) call timeMeasurement(.true.)

      ensemble1_loop : do ensemble1=1,numEnsembles
         ! first particle = real particle
         if (fullEnsemble) then
            ensemble2_Start=1
            ensemble2_End=numEnsembles
         else
            ensemble2_Start=ensemble1
            ensemble2_End=ensemble1
         end if
         if (debug) write(*,*) 'Loop über',ensemble1, ensemble2_Start, &
            &   ensemble2_End

         index1_loop: do index1=1,size_real_dim2
            part1=>partReal(ensemble1,index1)
            if (part1%ID < 0) exit index1_loop
            if (part1%ID <= 0) cycle index1_loop
            if (part1%ID.eq.photon) cycle index1_loop

            NULLIFY(deuterium_pertOrigin)
            deuterium_pertOrigin=>partReal(ensemble1,index1)
            deuterium_pertOrigin_flag=2
            ensemble2_loop : do ensemble2=ensemble2_Start,ensemble2_End
               ! second particle = perturbative
               index2_loop : do index2=1,size_Pert_dim2
                  part2 => partPert(ensemble2,index2)
                  if (part2%ID < 0) exit index2_loop
                  if (part2%ID <= 0) cycle index2_loop
                  if (part2%ID.eq.photon) cycle index2_loop
                  if (noRecollisions.and.(part2%lastCollTime>time-1e-6)) &
                       cycle index2_loop

                  part2S = part2

                  ! Check that they didn't collide just before
                  if (check_justCollided(part1,part2)) cycle index2_loop

                  ! No Nuk-Nuk collisions
                  if (noNucNuc .and. (part1%ID==nucleon .and. part2%ID==nucleon))&
                       cycle index2_loop

                  call resetNumberGuess

                  call collide_2body((/part1,part2/), finalState, time, flag, &
                       HiEnergyFlag, HiEnergyType, &
                       PauliIncluded_out=pauliIsIncluded)
                  if (.not.flag) cycle index2_loop
                  !          if (debug) write(*,*) 'final states.  &
                  !          & IDs=',finalState%ID,'  charges=',finalState%Charge
                  !          & if (debug) write(*,*) 'HiEnergy: ',HiEnergyFlag, &
                  !          & HiEnergyType

                  flag = finalCheck((/part1,part2/), finalState, HiEnergyFlag)
                  if (.not.flag) cycle index2_loop

                  ! Accept or reject the event randomly,
                  ! using the in-medium cross section ratios:
                  if (.not.accept_event((/part1,part2/),finalState,HiEnergyFlag)) cycle index2_loop

                  if (.not.pauliIsIncluded) flag = checkPauli(finalState,partReal)
                  if (.not.flag) cycle index2_loop

                  !           write(*,*) 'colliding:',part1%number,part2%number,&
                  !           & HiEnergyType

                  NumbersAlreadySet = AcceptGuessedNumbers()

                  if (DoJustAbsorptive) finalState%ID = 0

                  !simulate Effenberger treatment of Deltas
                  if (justDeleteDelta) then
                     if (((part1%ID.eq.nucleon) .and. (part2%ID.eq.delta)) &
                          .or.((part2%ID.eq.nucleon) .and. (part1%ID.eq.delta)))&
                        &   then
                        !check that we have N N in finalstate
                        justdeletedelta_count=0
                        justdeletedelta_countO=0
                        do i=1,maxout
                           if (finalState(i)%id <= 0) exit
                           if (finalstate(i)%id.eq.1) justdeletedelta_count=&
                              & justdeletedelta_count+1
                           if (finalstate(i)%id.ne.1) justdeletedelta_countO=&
                              & justdeletedelta_countO+1
                        end do
                        if (justdeletedelta_count.eq.2.and. &
                           & justdeletedelta_countO.eq.0) finalstate%id=0
                     end if
                  end if

                  call rate((/part1,part2/),finalState,time)
                  ! Compute various collision rates

                  ! (1) Label event by eventNumber, such that we can track
                  ! the perturbative particle to its
                  ! real scattering partner.

                  number = pert_numbering(part1)
                  if (part2%firstEvent==0) then
                     number2=pert_firstnumbering(part1,part2)
                  else
                     number2=part2%firstEvent
                  end if

                  do i=1,maxout
                     if (finalState(i)%id <= 0) exit

                     finalState(i)%event = number
                     finalState(i)%lastCollTime = time
                     finalState(i)%perweight = part2%perweight
                     finalState(i)%firstEvent = number2
                     finalState(i)%pert = .true.
                     if (.not.NumbersAlreadySet) call setNumber(finalState(i))

                     !#########################################################
                     ! Writing statistical information
                     if (useStatistics .and. finalState(i)%ID==pion) then
                        call saveInfo(finalState(i)%number,ensemble2, &
                             & finalstate(i)%pos, part1%ID,part2%ID,0)
                        if (debugSaveInfo) write(99,*) finalState(i)%number, 6,&
                           &ensemble2,finalstate(i)%pos, part1%ID,part2%ID,0
                     end if
                     if (printPositions .and. (finalState(i)%id==pion .or. &
                        & finalState(i)%id==nucleon)) &
                        & call positionPrinter(finalstate(i))
                     !#########################################################
                  end do

                  ! (2) Eliminate incoming perturbative particles
                  partPert(ensemble2,index2)%Id=0

                  ! (3) Create final state particles
                  posOut = 0
                  if (particlePropagated(finalState(1))) then
                     partPert(ensemble2,index2) = finalState(1)
                     posOut(1:2,1) = (/ensemble2,index2/)
                  end if

                  if (fullensemble) then
                     call setIntoVector(finalState(2:),partPert,setFlag,  &
                          & .true.,positions=posOut(:,2:))
                  else
                     call setIntoVector(finalState(2:),   &
                          &partPert(ensemble1:ensemble1,:),setFlag, &
                          & .true.,positions=posOut(:,2:))
                     posOut(1,2:) = ensemble1
                  end if

                  call ReportEventNumber((/part1,part2S/), finalState,  &
                       finalstate(1)%event, time, 212, HiEnergyType, &
                       iEns1=ensemble1, iEns2=ensemble2)

                  call CollHist_UpdateHist((/part1,part2S/), finalState, &
                       & (/ensemble2,index2/), posOut, finalState(1)%perweight)

                  ! (4) Check that setting into perturbative particle vector
                  ! worked out
                  if (.not.setFlag) then
                     write(*,*) 'Perturbative particle vector too small!'
                     write(*,*) size(finalState),  lbound(partPert), &
                                & ubound(partPert)
                     write(*,*) 'Dumping real particles to files  &
                                & "RealParticles_stop_*!"'
                     call WriteParticleVector("RealParticles_stop",partReal)
                     write(*,*) 'Dumping perturbative particles to files &
                                & "PertParticles_stop_*!"'
                     call WriteParticleVector("PertParticles_stop",partPert)
                     call Traceback('collision Term, two-body')
                  end if

                  ! Add a hole in the target nucleus (analysis only):
                  call ResidueAddPH(number2,part1)

               end do index2_loop ! Index2 of second particle
            end do ensemble2_loop ! Ensemble2 of second particle
         end do index1_loop! Index1 of first particle
      end do ensemble1_loop! Ensemble1 of first particle
      if (printPositions) call positionPrinter()

      if (debug) call timeMeasurement(.false.)
      nullify(deuterium_pertOrigin)
      deuterium_pertOrigin_flag=999


    end subroutine realPert

  end subroutine twoBody


  !****************************************************************************
  !****s* collisionTerm/twoBody_local
  ! NAME
  ! subroutine twoBody_local(partPert, partReal, time)
  !
  ! PURPOSE
  ! * Administrates the 2-body processes.
  ! * The collision criteria is based upon the local collision criteria.
  !   Therefore the volume elements (module VolumeElements) must be
  !   initialized. Only scatterings within one volume element are allowed.
  !
  ! INPUTS
  ! * type(particle),intent(inOUT),target,dimension(:,:) ::
  !   partReal -- real particles
  ! * type(particle),intent(inOUT),target,dimension(:,:) ::
  !   partPert -- perturbative particles
  ! * real, intent(in) :: time
  !
  ! OUTPUT
  ! * partPert and partReal are changed
  !****************************************************************************
  subroutine twoBody_local(partPert, partReal, time)

    use master_2Body, only: collide_2body
    use particleDefinition
    use output, only: WriteParticleVector, timeMeasurement
    use IdTable, only: nucleon, delta, pion, photon
    use VolumeElements, only: &
         & VolumeElements_CLEAR_Pert, VolumeElements_CLEAR_Real, &
         & VolumeElements_SETUP_Pert, VolumeElements_SETUP_Real
    use XsectionRatios, only: accept_event
    use pauliBlockingModule, only: checkPauli
    use insertion, only: particlePropagated, setIntoVector
    use collisionTools, only: finalCheck
    use inputGeneral, only: freezeRealParticles
    use residue, only: ResidueAddPH

    type(particle), intent(inout), dimension(:,:) :: partPert
    type(particle), intent(inout), dimension(:,:), target :: partReal
    real, intent(in) :: time

    integer :: iInd1,iInd2,iEns1,iEns2
    type(particle), dimension(1:maxOut) :: finalState ! final state of particles
    logical :: Flag, setFlag, HiEnergyFlag
    integer :: size_real_dim2!,size_pert_dim2
    integer :: HiEnergyType,i,number
    real :: XS
    integer, save :: justdeletedelta_count=0, justdeletedelta_countO=0
    logical, save :: firstCall = .true.

    size_real_dim2=size(partReal,dim=2)
    !size_pert_dim2=size(partPert,dim=2)

    !--- real<->real collisions:

    ! note: freezeRealParticles and twoBodyP..RealReal are exclusive
    if (firstCall) then
       if (freezeRealParticles) then ! do it once
          call VolumeElements_CLEAR_Real
          call VolumeElements_SETUP_Real(partReal)
       end if
       firstCall = .false.
    end if
    if (twoBodyProcessesRealReal) then
       call VolumeElements_CLEAR_Real
       if (noRecollisions) then
          call VolumeElements_SETUP_Real(partReal,time)
       else
          call VolumeElements_SETUP_Real(partReal)
       end if
       call realReal
    end if

    !--- real<->pert collisions:

    call VolumeElements_CLEAR_Pert
    if (twoBodyProcessesRealPert) then
       ! if RealReal happened, we have to rebuild tVE_Real!
       if (.not.freezeRealParticles) then
          call VolumeElements_CLEAR_Real
          call VolumeElements_SETUP_Real(partReal)
       end if
       call VolumeElements_SETUP_Pert(partPert)
       call realPert
    end if

  contains

    !**************************************************************************
    ! real<->real collisions
    !**************************************************************************

    subroutine realReal
      use VolumeElements, only: VolumeElements_InitGetPart_RealReal, &
          & VolumeElements_GetPart_RealReal
      use collisionNumbering, only: check_justCollided, real_numbering,  &
          &real_firstnumbering, ReportEventNumber
      use twoBodyStatistics, only: rate

      type(particle), pointer :: part1, part2

      logical :: numbersAlreadySet, pauliIsIncluded
      integer :: nRealPart, weight
      integer, dimension(2,maxOut) :: posOut

      if (debug) write(*,*) 'real<->real'
      if (debug) call timeMeasurement(.true.)

      call VolumeElements_InitGetPart_RealReal

      do
         if (.not. VolumeElements_GetPart_RealReal(part1, part2, nRealPart, &
            & iEns1,iInd1,iEns2,iInd2)) exit

         weight = (nRealPart-1)+mod(nRealPart,2)
         ! ??? should be n for n odd and (n-1) for n even ???
         ! ??? max number of pairs: n*(n-1)/2, i.e. 15,21 for n=6,7
         ! ??? number of selected pairs: n/2 (integer division), i.e 3,3 for n=6,7

         ! Check that they didn't collide just before
         if (check_justCollided(part1,part2)) cycle

         ! No Nuk-Nuk collisions
         if (noNucNuc .and. (part1%ID==nucleon) .and. (part2%ID==nucleon)) cycle

         ! No Photon-Particle Collisions
         if ((part1%ID==photon) .or. (part2%ID==photon)) cycle

         call resetNumberGuess
         call collide_2body((/part1,part2/), finalState, time, flag, &
                           & HiEnergyFlag, HiEnergyType, &
                           & weightLocal=weight, sigTot_out=XS, &
                           & PauliIncluded_out=pauliIsIncluded)
         ! weightLocal=Number of real particles in volume, integer

         if (.not.flag) cycle

         flag = finalCheck ((/part1,part2/), finalState, HiEnergyFlag)
         if (.not.flag) cycle

         ! Accept or reject the event randomly,
         ! using the in-medium cross section ratios:
         if (.not.accept_event((/part1,part2/),finalState,HiEnergyFlag)) cycle

         if (.not.pauliIsIncluded) flag = checkPauli(finalState,partReal)
         if (.not.flag) cycle

         call rate((/part1,part2/),finalState,time)
         ! Compute various collision rates

         ! (1) Label event by eventNumber
         number=real_numbering()
         finalState%event(1)=number
         finalState%event(2)=number

         finalState%lastCollTime = time
         finalState%pert = .false.
         finalState%perweight = min(Part1%perweight, Part2%perweight)
         number = min(part1%firstEvent,part2%firstEvent)
         if (number==0) number = real_firstnumbering()
         finalState%firstEvent=number

         call ReportEventNumber ((/part1,part2/), finalState,  &
              finalState(1)%event, time, 211, HiEnergyType, &
              iEns1=iEns1, iEns2=iEns2)

         ! Add a hole in the target nucleus (analysis only):
         if(part1%event(1).lt.1000000) then
            call ResidueAddPH(part1%firstEvent,part1)
         else if(part2%event(1).lt.1000000) then
            call ResidueAddPH(part2%firstEvent,part2)
         end if

         ! (2) Create final state particles
         NumbersAlreadySet = AcceptGuessedNumbers()

         posOut = 0

         if (particlePropagated(finalState(1))) then
            if (.not.NumbersAlreadySet) call setNumber(finalState(1))
            part1 = finalState(1)
            posOut(1:2,1) = (/iEns1,iInd1/)
         else
            part1%Id=0
         end if

         if (particlePropagated(finalState(2))) then
            if (.not.NumbersAlreadySet) call setNumber(finalState(2))
            part2 = finalState(2)
            posOut(1:2,2) = (/iEns2,iInd2/)
         else
            part2%Id=0
         end if

         call setIntoVector(finalState(3:), partReal, setFlag, &
              NumbersAlreadySet, positions=posOut(:,3:))

         ! (3) Check that setting into real particle
         !     vector worked out :
         if (.not. setFlag) then
            write(*,*) 'Real particle vector too small!'
            write(*,*) size(finalState),&
                 lbound(partReal), ubound(partReal)
            write(*,*) 'Dumping real particles to files "RealParticles_stop_*!"'
            call WriteParticleVector("RealParticles_stop",partReal)
            write(*,*) 'Dumping perturbative particles to files &
                       & "PertParticles_stop_*!"'
            call WriteParticleVector("PertParticles_stop",partPert)
            call Traceback('collision Term, two-body')
         end if

      end do

    end subroutine realReal

    !**************************************************************************
    ! real<->pert collisions
    !**************************************************************************
    subroutine realPert
      use statistics, only: saveInfo
      use VolumeElements, only: VolumeElements_InitGetPart_RealPert, &
          & VolumeElements_GetPart_RealPert
      use CollHistory, only: CollHist_UpdateHist
      use collisionNumbering, only: check_justCollided, pert_numbering, &
          & pert_firstnumbering, ReportEventNumber
      use twoBodyStatistics, only: rate

      logical :: numbersAlreadySet, pauliIsIncluded
      integer :: nRealPart, number2
      !real :: dummy
      integer, dimension(2,maxOut) :: posOut
      type(particle), pointer :: part1, part2
      type(particle) :: partS2

      if (debug) then
         write(*,*) 'real<->pert'
         call timeMeasurement(.false.)
         call timeMeasurement(.true.)
      end if

      call VolumeElements_InitGetPart_RealPert

      do
         if (.not. VolumeElements_GetPart_RealPert(part1, part2, nRealPart, &
              iEns1,iInd1, iEns2,iInd2)) exit

         ! Check that they didn't collide just before
         if (check_justCollided(part1,part2)) cycle

         ! No Nuk-Nuk collisions
         if (noNucNuc .and. (part1%ID==nucleon) .and. (part2%ID==nucleon)) cycle

         ! No Photon-Particle Collisions
         if ((part1%ID==photon) .or. (part2%ID==photon)) cycle

         call resetNumberGuess
         call collide_2body((/part1,part2/), finalState, time, flag, &
              & HiEnergyFlag, HiEnergyType, &
              & weightLocal=nRealPart, sigTot_out=XS, &
              & PauliIncluded_out=pauliIsIncluded)

         ! weightLocal=Number of real particles in volume, integer

         if (.not.flag) cycle
!         if (debug) write(*,*) 'final states. IDs=',finalState%ID, &
!            &'  charges=',finalState%Charge
!         if (debug) write(*,*) 'HiEnergy: ',HiEnergyFlag,HiEnergyType

         flag = finalCheck((/part1,part2/), finalState, HiEnergyFlag)
         if (.not.flag) cycle

         ! Accept or reject the event randomly,
         ! using the in-medium cross section ratios:
         if (.not.accept_event((/part1,part2/),finalState,HiEnergyFlag)) cycle

         if (.not.pauliIsIncluded) flag = checkPauli(finalState,partReal)

!!$         if (part2%ID.eq.101) then
!!$            dummy = sqrtS((/part1,part2/))
!!$
!!$            if (flag) then
!!$               call AddHist(hXStot,dummy,1.0,1.0)
!!$               if (HiEnergyType.eq.-1) call AddHist(hXSelast,dummy,1.0,1.0)
!!$            else
!!$               call AddHist(hXStot,dummy,1.0,0.0)
!!$               if (HiEnergyType.eq.-1) call AddHist(hXSelast,dummy,1.0,0.0)
!!$            end if
!!$!            write(*,*) part2%charge,part1%charge, HiEnergyType,flag,dummy
!!$         end if

         if (.not.flag) cycle

!         write(*,*) 'colliding:',part1%number,part2%number,HiEnergyType

         NumbersAlreadySet = AcceptGuessedNumbers()

         if (DoJustAbsorptive) finalState%ID = 0

         !simulate Effenberger treatment of Deltas
         if (justDeleteDelta) then
            if (((part1%ID.eq.nucleon) .and. (part2%ID.eq.delta)).or. &
               & ((part2%ID.eq.nucleon) .and. (part1%ID.eq.delta))) then
               !check that we have N N in finalstate
               justdeletedelta_count=0
               justdeletedelta_countO=0
               do i=1,maxout
                  if (finalState(i)%id <= 0) exit
                  if (finalstate(i)%id.eq.1) &
                  & justdeletedelta_count=justdeletedelta_count+1
                  if (finalstate(i)%id.ne.1) &
                     &justdeletedelta_countO=justdeletedelta_countO+1
               end do
               if (justdeletedelta_count.eq.2.and.justdeletedelta_countO.eq.0) &
                  & finalstate%id=0
            end if
         end if

         ! Compute various collision rates:
         call rate((/part1,part2/),finalState,time)

         ! (1) Label event by eventNumber, such that we can track
         ! the perturbative particle to its real scattering partner.

         number=pert_numbering(part1)
         if (part2%firstEvent==0) then
            number2=pert_firstnumbering(part1,part2)
         else
            number2=part2%firstEvent
         end if

         do i=1,maxout
            if (finalState(i)%id <= 0) exit

            finalState(i)%event = number
            finalState(i)%lastCollTime = time
            finalState(i)%perweight = part2%perweight
            finalState(i)%firstEvent = number2
            finalState(i)%pert = .true.
            if (.not.NumbersAlreadySet) call setNumber(finalState(i))

            !##################################################################
            ! Writing statistical information
            if (useStatistics .and. finalState(i)%ID==pion) then
               call saveInfo(finalState(i)%number,iEns2,finalstate(i)%pos, &
                    & part1%ID,part2%ID,0)
               if (debugSaveInfo) write(99,*) finalState(i)%number,  &
                  & iEns2,finalstate(i)%pos, part1%ID,part2%ID,0
            end if
            if (printPositions .and. (finalState(i)%id==pion .or. &
               & finalState(i)%id==nucleon)) &
                 call positionPrinter(finalstate(i))
            !##################################################################
         end do

         ! (2) Eliminate incoming perturbative particles
         partS2 = part2
         part2%Id = 0

         ! (3) Create final state particles
         posOut = 0
         if (particlePropagated(finalState(1))) then
            part2 = finalState(1)
            posOut(1:2,1) = (/iEns2,iInd2/)
         end if

         call setIntoVector(finalState(2:),partPert,setFlag,.true., &
              & positions=posOut(:,2:))

         call ReportEventNumber((/part1,partS2/), finalState,  &
              finalstate(1)%event, time, 212, HiEnergyType, &
              iEns1=iEns1, iEns2=iEns2)

         call CollHist_UpdateHist((/part1,partS2/), finalState, &
              & (/iEns2,iInd2/), posOut, finalState(1)%perweight)

         ! (4) Check that setting into perturbative particle vector worked out
         if (.not.setFlag) then
            write(*,*) 'Perturbative particle vector too small!'
            write(*,*) size(finalState),  lbound(partPert), ubound(partPert)
            write(*,*) 'Dumping real particles to files "RealParticles_stop_*!"'
            call WriteParticleVector("RealParticles_stop",partReal)
            write(*,*) 'Dumping perturbative particles to files &
                       & "PertParticles_stop_*!"'
            call WriteParticleVector("PertParticles_stop",partPert)
            call Traceback('collision Term, two-body')
         end if

         ! Add a hole in the target nucleus (analysis only):
         call ResidueAddPH(number2,part1)

      end do

      if (printPositions) call positionPrinter()

      if (debug) call timeMeasurement(.false.)
    end subroutine realPert

  end subroutine twoBody_local


  !****************************************************************************
  !****s* collisionTerm/positionPrinter
  ! NAME
  ! subroutine positionPrinter(p)
  ! PURPOSE
  ! This routine saves the position of a given particle into a 2D Histogram as a
  ! function of cylinder coordinates z and rho. Each new entry is added to the
  ! previous ones.
  ! The histograms are meant to be used as tables of all production points.
  !
  ! If no input is given, the histogrames are printed to file.
  !
  ! INPUTS
  ! * type(particle), intent(in), optional :: p
  !
  ! OUTPUT
  ! * file 'ProdPlaces_piPlus.dat'    -- All saved positions of pi^+
  ! * file 'ProdPlaces_piNull.dat'    -- All saved positions of pi^0
  ! * file 'ProdPlaces_piMinus.dat'   -- All saved positions of pi^-
  ! * file 'ProdPlaces_nucleon.dat'   -- All saved positions of nucleons
  !****************************************************************************
  subroutine positionPrinter(p)
    use particleDefinition
    use hist2D
    use idTable, only: pion, nucleon

    type(particle), intent(in), optional :: p

    ! store position information in terms of z and the zylinder coordinate rho :
    type(histogram2D),save :: positions_pi(-1:1),positions_nuc

    logical, save :: firstCall=.true.
    real, dimension(1:3) :: r
    real :: rho, weight

    if (firstCall) then
       call createHist2d(positions_pi(1),  'ProdPlaces_pion_plus',&
            &  (/-8.,0./),(/8.,8./),(/0.25,0.25/))
       call createHist2d(positions_pi(0),  'ProdPlaces_pion_null',&
            & (/-8.,0./),(/8.,8./),(/0.25,0.25/))
       call createHist2d(positions_pi(-1), 'ProdPlaces_pion_minus',&
            & (/-8.,0./),(/8.,8./),(/0.25,0.25/))
       call createHist2d(positions_nuc,    'ProdPlaces_nucleon',   &
            & (/-8.,0./),(/8.,8./),(/0.25,0.25/))
       firstCall=.false.
    end if

    if (present(p)) then
       r = p%pos
       rho = sqrt(r(1)**2+r(2)**2)
       weight = p%perweight
       select case (p%ID)
       case (pion)
          call AddHist2D(positions_pi(p%charge),(/r(3),rho/),weight)
       case (nucleon)
          call AddHist2D(positions_nuc,(/r(3),rho/),weight)
       end select
       return
    end if

    ! no input, therefore produce output:
    call WriteHist2D_Gnuplot(positions_pi(1),file='ProdPlaces_piPlus.dat')
    call WriteHist2D_Gnuplot(positions_pi(0),file='ProdPlaces_piNull.dat')
    call WriteHist2D_Gnuplot(positions_pi(-1),file='ProdPlaces_piMinus.dat')
    call WriteHist2D_Gnuplot(positions_nuc,file='ProdPlaces_nucleon.dat')


  end subroutine positionPrinter


  !****************************************************************************
  !****s* collisionTerm/threeBody
  ! NAME
  ! subroutine threeBody(partPert, partReal, time)
  ! PURPOSE
  ! Administrates the 3-body processes.
  !
  ! INPUTS
  ! * type(particle),dimension(:,:) :: partPert -- perturbative particles
  ! * type(particle),dimension(:,:) :: partReal -- real particles
  ! * real                          :: time     -- actual time step
  !
  ! OUTPUT
  ! * partPert and partReal are changed
  !****************************************************************************
  subroutine threeBody(partPert, partReal, time)

    use IDTable, only: pion, nucleon, Delta
    use inputGeneral, only: fullEnsemble,localEnsemble,freezeRealParticles
    use particleDefinition
    use master_3Body, only: GetRadiusNukSearch, nukSearch, make_3Body_Collision
    use collisionNumbering, only: pert_numbering,real_numbering,  &
        & ReportEventNumber,pert_firstnumbering,real_firstnumbering
    use output, only: WriteParticleVector
    use VolumeElements, only: VolumeElements_NukSearch, &
        & VolumeElements_CLEAR_Pert, VolumeElements_CLEAR_Real, &
        & VolumeElements_SETUP_Pert, VolumeElements_SETUP_Real !,VolumeElements_Statistics
    use CollHistory, only: CollHist_UpdateHist
    use pauliBlockingModule, only: checkPauli
    use insertion, only: particlePropagated, setIntoVector
    use twoBodyStatistics, only: rate
    use collisionTools, only: finalCheck
    use residue, only: ResidueAddPH

    type(particle), intent(inout), dimension(:,:), target :: partPert, partReal
    real, intent(in) :: time

    INTEGER :: i,j,number,number2,ii
    logical :: flagOk
    logical, parameter :: nukSearch_VE=.true.
    type(particle), pointer:: proton1, proton2              ! Closest protons
    type(particle), pointer:: neutron1, neutron2            ! Closest neutrons
    !  ! Particles which one is scattering with:
    type(particle), pointer:: part, scatterPartner1, scatterPartner2
    type(particle), dimension(1:3) :: FinalState
    ! logical :: justDeleteDelta=.false.! If Delta shall be just deleted -
    !                             ! therefore energy conservation is violated.
    integer, dimension(2,maxOut) :: posOut
    type(particle) :: partS2

    logical, save :: firstCall = .true.

    if (localEnsemble) then

       if (firstCall.or.(.not.freezeRealParticles)) then
          call VolumeElements_CLEAR_Real
          call VolumeElements_SETUP_Real(partReal)
       end if
       call VolumeElements_CLEAR_Pert
       call VolumeElements_SETUP_Pert(partPert)

       !       call VolumeElements_Statistics
    end if
    firstCall = .false.

    !**************************************************************************
    ! perturbative Particle with two real ones
    !**************************************************************************

    do i=lbound(partPert,dim=1),ubound(partPert,dim=1)
       do j=lbound(partPert,dim=2),ubound(partPert,dim=2)
          part => partPert(i,j)

          if (part%ID.ne.pion.and.part%ID.ne.delta) cycle

          if (associated(proton1))  Nullify(proton1)
          if (associated(proton2))  Nullify(proton2)
          if (associated(neutron1)) Nullify(neutron1)
          if (associated(neutron2)) Nullify(neutron2)

          ! Find closest nucleons
          if (fullensemble) then
             if (nukSearch_VE.and.localEnsemble) then
                call VolumeElements_NukSearch(part, GetRadiusNukSearch(), &
                     proton1,proton2,neutron1,neutron2, flagOk)
             else
                call NukSearch(part,partReal,&
                     proton1,proton2,neutron1,neutron2, flagOk)
             end if
          else
             call NukSearch(part,partReal(i:i,:),&
                  proton1,proton2,neutron1,neutron2, flagOk)
          end if

          if (.not.flagOk) cycle
          call make_3Body_Collision(part, proton1,proton2,neutron1,neutron2, &
               scatterPartner1,scatterPartner2,finalstate, flagOk)

          if (.not.flagOk) cycle

          if (part%ID==Delta .and. JustDeleteDelta) then
             ! just delete the delta, no final state
             finalState%event(1)=pert_numbering(scatterPartner1)
             finalState%event(2)=pert_numbering(scatterPartner2)
             call ReportEventNumber((/part,scatterPartner1,scatterPartner2/), &
                  finalState, finalState(1)%event,time,3112)
             ! (2) Eliminate incoming perturbative particles
             part%Id=0
             cycle
          end if

          flagOk = finalCheck((/part,scatterPartner1,scatterPartner2/), &
               finalState, .false.)
          if (.not.flagOk) cycle
          if (part%ID/=Delta) then
             flagOk = checkPauli(finalState,partReal)
          else
             ! Pauli Blocking is already included in the delta "decay width",
             ! therefore we should not count it twice!
             flagOk = .true.
          end if
          if (.not.flagOk) cycle
          ! (1) Label event by eventNumber,
          ! such that we can track the perturbative particle to its
          ! real scattering partner.
          finalState%event(1)=pert_numbering(scatterPartner1)
          finalState%event(2)=pert_numbering(scatterPartner2)

          finalState%lastCollTime = time
          finalState%perweight = part%perweight
          finalState%pert = .true.

          if (part%firstEvent==0) then
             number2 = pert_firstnumbering(scatterPartner1,part)
          else
             number2 = part%firstEvent
          end if
          finalState%firstEvent = number2

          do ii=1,size(finalState)
             if (particlePropagated(finalState(ii))) &
                  call setNumber(finalState(ii))
          end do

          call ReportEventNumber((/part,scatterPartner1,scatterPartner2/), &
               finalState, finalState(1)%event, time, 3112)

          call rate((/part,scatterPartner1,scatterPartner2/), &
               finalState, time)

          if(scatterPartner1%event(1).lt.1000000) &
               call ResidueAddPH(number2,scatterPartner1)
          if(scatterPartner2%event(1).lt.1000000) &
               call ResidueAddPH(number2,scatterPartner2)

          ! (2) Eliminate incoming perturbative particles
          partS2 = part
          part%Id = 0

          ! (3) Create final state particles
          posOut = 0
          if (particlePropagated(finalState(1))) then
             partPert(i,j) = finalState(1)
             posOut(1:2,1) = (/i,j/)
          end if
          if (fullensemble) then
             call setIntoVector(finalState(2:), partPert, flagOk, .true., &
                               & positions=posOut(:,2:))
          else
             call setIntoVector(finalState(2:), partPert(i:i,:), flagOk, &
                               & .true., positions=posOut(:,2:))
             posOut(1,2:) = i
          end if

          call CollHist_UpdateHist((/partS2,scatterPartner1,scatterPartner2/), &
                  & finalState, (/i,j/), posOut, finalState(1)%perweight)

          ! (4) Check that setting into perturbative particle vector worked out
          if (.not.flagOk) then
             write(*,*) 'Perturbative particle vector too small!'
             write(*,*) size(finalState),  lbound(partPert), ubound(partPert)
             write(*,*) 'Dumping real particles to files "RealParticles_stop_*!"'
             call WriteParticleVector("RealParticles_stop",partReal)
             write(*,*) 'Dumping perturbative particles to files &
                        & "PertParticles_stop_*!"'
             call WriteParticleVector("PertParticles_stop",partPert)
             call Traceback('collision Term, three-body')
          end if
       end do
    end do

    !**************************************************************************
    ! real with two real ones
    !**************************************************************************

    do i=lbound(partReal,dim=1), ubound(partReal,dim=1)
       do j=lbound(partReal,dim=2), ubound(partReal,dim=2)
          part => partReal(i,j)

          if (part%ID/=pion .and. part%ID/=delta) cycle

          if (associated(proton1))  Nullify(proton1)
          if (associated(proton2))  Nullify(proton2)
          if (associated(neutron1)) Nullify(neutron1)
          if (associated(neutron2)) Nullify(neutron2)

          ! Find closest nucleons
          if (fullensemble) then
             if (nukSearch_VE.and.localEnsemble) then
                call VolumeElements_NukSearch(part, &
                     GetRadiusNukSearch(),&
                     proton1,proton2,neutron1,neutron2, flagOk)
             else
                call NukSearch(part, partReal, &
                     proton1, proton2, neutron1, neutron2, flagOk)
             end if
          else
             call NukSearch(part, partReal(i:i,:), &
                  proton1, proton2, neutron1, neutron2, flagOk)
          end if

!!$          if (.not.flagOk) then
!!$             write(*,*) 'nothing found'
!!$          else
!!$             write(*,*) '---- found'
!!$          end if


          if (.not.flagOk) cycle
          call make_3Body_Collision(part, proton1,proton2,neutron1,neutron2, &
               scatterPartner1, scatterPartner2, finalstate, flagOk)
          if (.not.flagOk) cycle

          if (part%ID==Delta .and. JustDeleteDelta) then
             number = real_numbering()
             call ReportEventNumber((/part,scatterPartner1,scatterPartner2/), &
                  finalState, (/number,number/), time, 3111)
             ! convert the Delta to a nucleon:
             part%Id = nucleon
             ! dont care about total charge:
             part%charge = (part%charge + 1)/2
             part%event = number
             cycle
          end if

          flagOk = finalCheck((/part,scatterPartner1,scatterPartner2/), &
               finalState, .false.)
          if (.not.flagOk) cycle
          if (part%ID/=Delta) then
             flagOk = checkPauli(finalState,partReal)
          else
             ! Pauli Blocking is already included in the delta "decay width",
             ! therefore we should not count it twice!
             flagOk = .true.
          end if
          if (.not.flagOk) cycle

          ! (1) Label event by eventNumber
          number = real_numbering()
          finalState%event(1) = number
          finalState%event(2) = number

          finalState%lastCollTime = time
          finalState%pert = .false.
          finalState%perweight = min(part%perweight, &
               scatterPartner1%perweight, scatterPartner2%perweight)
          number2 = min(part%firstEvent, &
               scatterPartner1%firstEvent, scatterPartner2%firstEvent)
          if (number2==0) number2 = real_firstnumbering()
          finalState%firstEvent = number2

          call ReportEventNumber((/part,scatterPartner1,scatterPartner2/), &
               finalState, finalState(1)%event, time, 3111)

          call rate((/part,scatterPartner1,scatterPartner2/), &
               finalState, time)

          ! Add two holes in the target nucleus:
          if(scatterPartner1%event(1).lt.1000000) &
               call ResidueAddPH(number2,scatterPartner1)
          if(scatterPartner2%event(1).lt.1000000) &
               call ResidueAddPH(number2,scatterPartner2)

          ! (2) Eliminate incoming particles
          part%Id = 0
          scatterPartner1%Id = 0
          scatterPartner2%Id = 0

          ! (3) Create final state particles
          if (particlePropagated(finalState(1))) then
             call setNumber(finalState(1))
             scatterPartner1 = finalState(1)
          end if
          if (particlePropagated(finalState(2))) then
             call setNumber(finalState(2))
             scatterPartner2 = finalState(2)
          end if
          if (particlePropagated(finalState(3))) then
             call setNumber(finalState(3))
             partReal(i,j) = finalState(3)
          end if
       end do
    end do

  end subroutine threeBody


end module collisionTerm
