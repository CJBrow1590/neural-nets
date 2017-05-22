#!/bin/sh -f

COAT="/Users/harrison/coatjava-builds/clas12-offline-software/coatjava/"
classPath="$COAT/lib/services/*:$COAT/lib/clas/*:$COAT/lib/utils/*:src/"

javac -cp $classPath src/physics/Nid.java

javac -cp $classPath src/physics/Narticle.java

javac -cp $classPath src/analysis/Generator.java

java -Xmx1536m -Xms1024m -cp $classPath analysis.Generator 200000
