program Emiss_FinalEvents

  implicit none


  integer :: iRun, iEv, ID, IQ, hist, prodID
  double precision :: weight, enu, weight0
  double precision, dimension(1:3) :: pos
  double precision, dimension(0:3) :: mom

  integer :: iRun0, iEv0, prodID0, iPart, ios, nUnstable, nBound, iBin

  double precision, dimension(0:3) :: momIn, momOut

  double precision, dimension(-700:700) :: arr

  character(len=300) :: line
  logical :: doIt

  read (*,'(A)',iostat=ios) line

  arr = 0.
  iRun = 0
  iEv = 0
  doIt = .false.
  momIn = 0.
  momOut = 0.
  prodID0 = 0
  iPart = 0
  weight0 = 0.
  nUnstable = 0
  nBound = 0
  do
     read(*,*,iostat=ios) iRun,iEv,ID,IQ,weight,pos,mom,hist,prodID,enu

     if (ios /= 0) doIt = .true.
     if (iRun.ne.iRun0) doIt = .true.
     if (iEv.ne.iEv0) doIt = .true.

     if (doIt) then
        write(*,'(2I7,1P,2(1X,4E14.6),0P,3I7)') &
             iRun0, iEv0, momIn, momIn-momOut,prodID0,nUnstable,nBound

        iBin = (momIn(0)-momOut(0))*1000
        if (abs(iBin)<700) arr(iBin) = arr(iBin)+weight0

        if (ios /= 0) then
           do iBin=-700,700
              write(13,*) iBin,arr(iBin)
           end do
           exit
        end if

        iRun0 = iRun
        iEv0 = iEv
        momIn = 0.
        momOut = 0.
        prodID0 = prodID
        iPart = 0
        weight0 = 0.
        nUnstable = 0
        nBound = 0
        doIt = .false.
     end if
     iPart = iPart+1

     select case (iPart)
     case (1) ! neutrino
        momIn = momIn + mom
     case (2) ! scattered lepton
        momOut = momOut + mom
     case (3) ! hit nucleon
        ! do nothing
     case (4) ! remnant
        momIn = momIn - mom
!        momIn(0) = momIn(0) + (40-(ID-1000))*0.938d0 ! approx
!        momIn(0) = momIn(0) + (40-(ID-1000))*0.910d0 ! approx

        write(12,*) mom(0),(40-(ID-1000))*0.938d0

     case default
        if (weight>0) momOut = momOut + mom
     end select

     if (weight>0) then
        if (weight > weight0) weight0 = weight

        select case (ID)
        case (1)
           if (mom(0)**2-sum(mom(1:3)**2) < 0.936**2) nBound = nBound+1
        case (32,33,101,109,110,111)
           ! stable, do nothing
        case (902)
           ! do nothing
        case default
           nUnstable = nUnstable+1
        end select
     end if


  end do


end program Emiss_FinalEvents
