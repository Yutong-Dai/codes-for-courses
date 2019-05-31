#pragma once

#include "pet.h"
#include <vector>
#include <algorithm>
#include <string>

class Rock: public Pet {

    struct Form {
      Form(std::string name, bool isSuper): formName_(name), isSuperDuper_(isSuper) {if(isSuper) superFormName_ = new std::string(name); };
      std::string superfy(std::string s) {std::transform(s.begin(), s.end(), s.begin(), ::toupper); return s;};

      std::string formName_;
      std::string* superFormName_ = NULL;
      bool isSuperDuper_;
    };

    public:
        Rock();
        ~Rock();
        bool isOld();
        void eat();
        void sleep();
        void speak();
        void harden();
        void fetch();
        void play();
        void growUp();

    private:
        bool hasAged_;
        bool timelineRevealed_;
        std::vector<Form*> allForms_;
};
