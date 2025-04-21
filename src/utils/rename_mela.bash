for file in ./data/mela/*_melanin.jpg; do
  [ -e "$file" ] || continue
  newname="${file%_melanin.jpg}_mela.jpg"
  mv "$file" "$newname"
done