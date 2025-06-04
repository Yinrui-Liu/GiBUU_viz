!******************************************************************************
!****m* /CallStack
! NAME
! module CallStack
!
! PURPOSE
! This module provides a wrapper around compiler-specific routines.
!******************************************************************************
module CallStack

  implicit none
  private

  public :: traceback, system_command, getCompiler

contains

#ifdef ifort
#define intel
#endif
#ifdef ifx
#define intel
#endif

  !****************************************************************************
  !****s* CallStack/TRACEBACK
  ! NAME
  ! subroutine TRACEBACK(string,user_exit_code)
  !
  ! PURPOSE
  ! Write out the call stack of the program, depending on the compiler.
  ! One currently gets a backtrace with the following compilers:
  ! * ifort (via the routine TRACEBACKQQ)
  ! * gfortran 4.7 or higher (via ABORT)
  ! * nvfortran (via ABORT and environment variable NVCOMPILER_TERM)
  ! See also the documentation of these routines in the respective compiler
  ! manual.
  !
  ! INPUTS
  ! * character*(*), optional :: string -- header line to write
  ! * integer, optional :: user_exit_code -- code whether return or stop
  !   program
  ! * By specifying a user exit code of -1, control returns to the calling
  !   program. Specifying a user exit code with a positive value requests that
  !   specified value be returned to the operating system. The default value
  !   is 0, which causes the application to abort execution.
  !****************************************************************************
  subroutine TRACEBACK(string,user_exit_code)
#ifdef intel
    use IFCORE
#endif

    character*(*), intent(in), optional:: string
    integer, intent(in), optional :: user_exit_code

    flush(6)

#ifdef intel
!#warning "compiling with ifort"

    if (present(string)) then
       if (present(user_exit_code)) then
          call TRACEBACKQQ(string=string,user_exit_code=user_exit_code)
       else
          call TRACEBACKQQ(string=string)
       end if
    else
       if (present(user_exit_code)) then
          call TRACEBACKQQ(user_exit_code=user_exit_code)
       else
          call TRACEBACKQQ()
       end if
    end if
    return
#endif

#ifdef __GFORTRAN__
!#warning "compiling with gfortran"
    if (present(string)) then
       write(*,'(A)') string
       flush(6)
    end if
    if (present(user_exit_code)) then
       if (user_exit_code.eq.-1) then
          write(*,*) '--- no call stack trace possible ---'
          return
       end if
    end if
    ! ABORT is a GNU extension and gives a backtrace with gfortran 4.7 and above
    call abort()
#endif

#ifdef __NVCOMPILER
!#warning "compiling with nvfortran"
    if (present(string)) then
       write(*,'(A)') string
       flush(6)
    end if
    if (present(user_exit_code)) then
       if (user_exit_code.eq.-1) then
          write(*,*) '--- no call stack trace possible ---'
          return
       end if
    end if

    ! the following only produces a trace output, if before starting
    ! GiBUU, one has set the environment variable NVCOMPILER_TERM by
    ! calling "export NVCOMPILER_TERM=trace"
    call abort()

    return
#endif


!#warning "compiling with unknown compiler"

    if (present(string)) then
       write(*,'(A)') string
       flush(6)
    end if

    write(*,*) '--- no call stack trace possible ---'

    if (present(user_exit_code)) then
       if (user_exit_code.eq.-1) return
       stop 123
    end if
    stop


  end subroutine TRACEBACK


  !****************************************************************************
  !****s* CallStack/SYSTEM_COMMAND
  ! NAME
  ! subroutine SYSTEM_COMMAND(string)
  !
  ! PURPOSE
  ! Runs a shell command, as given by the argument 'string'.
  !
  ! INPUTS
  ! * character*(*) :: string -- shell command to execute
  !
  ! NOTES
  ! This routine is compiler-dependent. If compiled with ifort, the
  ! Intel-specific function SYSTEMQQ is called. Otherwise the F08 intrinsic
  ! procedure EXECUTE_COMMAND_LINE is used.
  !****************************************************************************
  subroutine SYSTEM_COMMAND(string)
#ifdef intel
    use IFPORT
#endif

    character*(*), intent(in):: string

#ifdef intel
    logical :: res
    res = SYSTEMQQ(string)
#elif __GFORTRAN__
    call EXECUTE_COMMAND_LINE(string)
#else
    call SYSTEM(string)
#endif
  end subroutine SYSTEM_COMMAND

  !****************************************************************************
  !****s* CallStack/getCompiler
  ! NAME
  ! integer function getCompiler()
  !
  ! PURPOSE
  ! return a number indicating the used compiler
  !
  ! OUTPUT
  ! * -1 : unknown
  ! * 1 : ifort
  ! * 2 : gfortran
  ! * 3 : nvfortran
  ! * 4 : ifx
  !****************************************************************************
  integer function getCompiler()

#if defined(ifort)
    getCompiler = 1
#elif defined(__GFORTRAN__)
    getCompiler = 2
#elif defined(__NVCOMPILER)
    getCompiler = 3
#elif defined(ifx)
    getCompiler = 4
#else
    getCompiler = -1
#endif

  end function getCompiler


end module CallStack
