!******************************************************************************
!****m* /neutrino_IDTable
! NAME
! module neutrino_IDTable
!
! PURPOSE
! Here important constants for neutrino init are stored
!******************************************************************************
module neutrino_IDTable
  implicit none

  public

  ! values for flavor_ID:
  integer, parameter :: electron  = 1
  integer, parameter :: muon      = 2
  integer, parameter :: taulepton = 3

  ! values for XsectionMode:
  integer, parameter :: integratedsigma=0
  integer, parameter :: dsigmadcosthetadelepton=1
  integer, parameter :: dsigmadQ2delepton=2
  integer, parameter :: dsigmadcostheta=4
  integer, parameter :: dsigmadelepton=5
  integer, parameter :: dsigmaMC=6
  integer, parameter :: dsigmaMC_dW=7
  integer, parameter :: dsigmaMC_dQ2=3
  integer, parameter :: fixedGamma=8

  integer, parameter :: EXP_dSigmadEnu=10
  integer, parameter :: EXP_dsigmadcosthetadelepton=11
  integer, parameter :: EXP_dsigmadQ2delepton=12
  integer, parameter :: EXP_dsigmadcostheta=14
  integer, parameter :: EXP_dsigmadelepton=15
  integer, parameter :: EXP_dsigmaMC=16
  integer, parameter :: EXP_dsigmaMC_dW=17
  integer, parameter :: EXP_dsigmaMC_dQ2=13

  character*(*), dimension(0:8), parameter :: sXsectionMode = (/&
       "sigma            ", &
       "dsigma/dcost dE' ", &
       "dsigma/dQ2 dE'   ", &
       "dsigmaMC/dQ2     ", &
       "dsigma/dcost     ", &
       "dsigma/dE'       ", &
       "sigmaMC          ", &
       "dsigmaMC/dW      ", &
       "fixed photon     " /)

  ! values for nuExp:
  integer, parameter :: MiniBooNE = 1
  integer, parameter :: K2K = 2
  integer, parameter :: Minos =3
  
  ! Comment on further ID's:
  !
  ! for lepton (electron, neutrino) events
  ! the final states of the initial interaction
  ! are denoted by a parameter IP which runs, 
  ! first, over the final baryon states, up to 
  ! IP = 31, as defined in IdTable.f90.
  
  ! In addition there are more IPs defined
  ! for additional final states:
  
  ! One pion bg events with neutron out: 32
  ! One pion bg events with proton out: 33
  ! DIS events: 34
  ! 2p2h (MEC) events: 35
  ! 2p2h with outgoing Delta : 36
  ! Two pion bg events: 37
  ! 
  ! These numbers must be respected if one ever
  ! wanted to extend the list of nucleon resonances
  ! in neutrinoXsection and IdTable.

  ! values for onePion:
  ! * n for outgoing channels with neutron
  ! * p for outgoing channels with proton
  integer, parameter :: chOnePionN=32
  integer, parameter :: chOnePionP=33

  integer, parameter :: chDIS=34

  integer, parameter :: chQE2p2h = 35
  integer, parameter :: chDelta2p2h = 36

  integer, parameter :: chTwoPion = 37

end module neutrino_IDTable
