//----------------------------------------------------------------------
//                             RootTuple
//  Author:  David Hall
//  Date:    29th August 2012
//  Website: http://roottuple.hepforge.org/
//
//  This declares the RootTuple class.
//
//----------------------------------------------------------------------

#ifndef RootTuple_H
#define RootTuple_H

// Standard C++ libraries
#include <iostream>
#include <string>
#include <vector>

// ROOT libraries
#include <TFile.h>
#include <TTree.h>

#ifndef WITHPOS
#define WITHPOS 0
#endif

class RootTuple{
public:
    // Constructors and destructors
    RootTuple();
    RootTuple(std::string filename, std::string treename);
    ~RootTuple();

    // Public methods
    void Initialise();
    void AddEvent();
    void Write();
    void Close();
    void AddParticle(
        int barcode, int ID, int charge,
        int UID, int history,
        double px, double py, double pz,
        double e, double x, double y, double z,
        int event0, int event1, int first_event
    );
    void SetWeight(double weight);

    void SetDoubleBranch(std::string branchname, double *ptr);
    void SetFloatBranch(std::string branchname, float *ptr);
    void SetIntBranch(std::string branchname, int *ptr);
    void SetBoolBranch(std::string branchname, bool *ptr);

private:
    // Private methods
    void DeclareBranches();
    void FillBranches();
    void ClearVectors();

    // ROOT file objects and variables
    TFile *m_file;
    TTree *m_tree;
    std::string m_filename, m_treename;

    // Branch variables
    Double_t m_weight;
    std::vector<Int_t> m_barcode, m_ID, m_charge, m_UID, m_history;
    std::vector<Double_t> m_Px, m_Py, m_Pz, m_E;
    std::vector<Int_t> m_event0, m_event1, m_first_event;
#if WITHPOS
    std::vector<Double_t> m_x, m_y, m_z;
#endif
};

#endif
