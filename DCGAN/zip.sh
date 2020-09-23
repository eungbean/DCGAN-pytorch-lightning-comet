# git archive HEAD -o 2019314120_EungbeanLee_Code-2.zip
git archive --prefix ${PWD##*/}/ HEAD -o ../${PWD##*/}-$(date "+%Y.%m.%d-%H.%M.%S").zip