for f in images/*
do
	echo "Processing $f"
	zip "${f}.zip" "compressed/${f}"
done
