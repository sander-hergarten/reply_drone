
cd $PWD/crates/dataset_generation/assets && for dir in images_all/*/; do if [ -d "$dir" ]; then file=$(find "$dir" -type f | sort -R | head -n 1); if [ -n "$file" ]; then cp "$file" "textures/$(basename "$dir")_$(basename "$file")"; echo "Copied: $file -> textures/$(basename "$dir")_$(basename "$file")"; fi; fi; done
