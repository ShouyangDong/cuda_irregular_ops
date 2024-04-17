#/bin/bash

export MY_CONTAINER="bangtransformer_1116"
num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
echo $MY_CONTAINER
if [ 0 -eq $num ];then
#xhost +local:
docker run -e  DISPLAY=$DISPLAY --net=host --pid=host --ipc=host \
-v /tmp/.X11-unix:/tmp/.X11-unix \
--privileged \
-it \
-v /projs/AE/dongshouyang/:/projs/AE/dongshouyang/ \
-v /mnt/:/mnt/ \
-v /data/:/data/  \
-v /mm_data/:/mm_data/  \
-v /opt/data/:/opt/data/  \
-v /usr/bin/cnmon:/usr/bin/cnmon \
--name $MY_CONTAINER  \
yellow.hub.cambricon.com/bangtransformer/devel/x86_64/bangtransformer:v0.4.0-devel-x86_64-ubuntu18.04 \
/bin/bash
else
xhost local:root
docker start $MY_CONTAINER
docker exec --workdir=/projs/AE/dongshouyang -ti $MY_CONTAINER /bin/bash
fi
