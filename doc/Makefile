
cnn.html: cnn.md
	pandoc --standalone --toc \
	--mathml \
	--css lab.css \
	--css cnn.css \
	--from=markdown+smart+yaml_metadata_block+latex_macros --to=html \
	-o cnn.html cnn.md
