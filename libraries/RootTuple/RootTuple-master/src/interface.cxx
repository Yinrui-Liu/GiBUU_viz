//----------------------------------------------------------------------
//                             RootTuple
//  Author:  David Hall
//  Date:    29th August 2012
//  Website: http://roottuple.hepforge.org/
//
//  This is the interface implementation. It declares a global pointer
//  to an instance of the RootTuple class. This keeps track of the data
//  being written to file. It also defines how the subroutines use the
//  RootTuple class's methods.
//
//----------------------------------------------------------------------

#include "interface.h"
#include "RootTuple.h"

// Declare global pointer
RootTuple *eventHandler;

void rootinit(const char *filename, int lfilename)
{
    eventHandler = new RootTuple(strFtoC(filename, lfilename), "RootTuple");
    eventHandler->Initialise();
}//rootinit

void rootwrite()
{
    eventHandler->Write();
}//rootwrite

void rootclose()
{
    eventHandler->Close();
    delete eventHandler;
}//rootclose

void rootaddparticle(
    int barcode, int ID, int charge,
    int UID, int history,
    double px, double py, double pz,
    double e, double x, double y, double z,
    int event0, int event1, int first_event)
{
    eventHandler->AddParticle(
        barcode, ID, charge,
        UID, history,
        px, py, pz,
        e, x, y, z,
        event0, event1, first_event
    );
}//rootaddparticle

void rootaddevent(double weight)
{
    eventHandler->SetWeight(weight);
    eventHandler->AddEvent();
}//rootaddevent

void rootadddouble(double *ptr, const char* branchname, int lbranchname)
{
    eventHandler->SetDoubleBranch(strFtoC(branchname, lbranchname), ptr);
}//rootadddouble

void rootaddfloat(float *ptr, const char* branchname, int lbranchname)
{
    eventHandler->SetFloatBranch(strFtoC(branchname, lbranchname), ptr);
}//rootaddfloat

void rootaddint(int *ptr, const char* branchname, int lbranchname)
{
    eventHandler->SetIntBranch(strFtoC(branchname, lbranchname), ptr);
}//rootaddint

void rootaddbool(bool *ptr, const char* branchname, int lbranchname)
{
    eventHandler->SetBoolBranch(strFtoC(branchname, lbranchname), ptr);
}//rootaddbool

// Converts a FORTRAN string to a C++ string
std::string strFtoC(const char *str, int len)
{
    int tlen = 0;
    char tem;

    // Counts non-blank characters in a string str until a first blank character or the end of the string is met
    while (str[tlen] != ' ' && (tlen < len) && (tem = str[tlen++], tem));

    char *tstr = new char[tlen+1];
    strncpy(tstr, str, tlen);
    tstr[tlen] = '\0';
    std::string ret = std::string(tstr);
    delete [] tstr;
    return ret;
}//strFtoC
