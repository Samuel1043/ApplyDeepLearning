for file in ./*/*.mp4;do 
	ffmpeg -i $file -r 15 -vf scale=512:-1  $(echo $file| sed --expression='s/\.mp4/\.gif/g') 
done
