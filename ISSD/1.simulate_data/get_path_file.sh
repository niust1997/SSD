#! /bin/bash
########################################################
#	readfile.sh
# 	list the name of all file in the directory named by users
#       copy right @Huyang
#########################################################
path=$1
result_file=$2
rm $result_file
read_dir ()
{
    for file in `ls $1`
    do
    #echo $1"/"$file
        if [ -d $1"/"$file ]
        then
            read_dir $1"/"$file
        else
            echo $1"/"$file >> $result_file
        fi
    done
}
read_dir $path