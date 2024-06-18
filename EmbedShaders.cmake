# edited from https://stackoverflow.com/questions/11813271/embed-resources-eg-shader-code-images-into-executable-library-with-cmake
file(WRITE ${out} "")
# Collect input files
file(GLOB bins ${dir}/*)
# Iterate through input files
foreach(bin ${bins})
    # Get filename from file path
    string(REGEX MATCH "([^/]+)$" filename ${bin})
    # Remove file extension
    string(REGEX REPLACE "\\.[^.]*$" "" filename ${filename})
    # Replace filename spaces & dots for C compatibility
    string(REGEX REPLACE "\\.| |-" "_" filename ${filename})
    # Read hex data from file
    file(READ ${bin} filedata HEX)
    # Convert hex data for C compatibility
    string(REGEX REPLACE "([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])" "0x\\4\\3\\2\\1," filedata ${filedata})
    # Append data to output file
    file(APPEND ${out} "const uint32_t ${filename}_code[] = {${filedata}};\nconst unsigned ${filename}_size = sizeof(${filename}_code);\n")
endforeach()
