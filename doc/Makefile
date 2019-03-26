
instructions.html: instructions.md
	pandoc --standalone --toc \
	--mathml \
	--css ../../../2019S/courseweb/labs/lab.css \
	--css instructions.css \
	--from=markdown+smart+yaml_metadata_block+latex_macros --to=html \
	-o instructions.html instructions.md

instructions-toc.md: instructions.md
	pandoc --toc -o instructions-toc.md instructions.md
