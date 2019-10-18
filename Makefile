index.html: README.md _static/head.html
	cat _static/head.html > index.html
	echo '<body role="document">' >> index.html
	echo '<div class="content">' >> index.html
	pandoc -f markdown -t html README.md >> index.html
	echo '</div>' >> index.html
	echo "</body>" >> index.html

release: index.html
	aws s3 --profile amazonai-events sync . s3://iccv19.mxnet.io/ --exclude 'slides/*' --exclude '.git*' --exclude 'Makefile' --grants read=uri=http://acs.amazonaws.com/groups/global/AllUsers
