# edited from https://stackoverflow.com/questions/11813271/embed-resources-eg-shader-code-images-into-executable-library-with-cmake

# Collect input files
file(GLOB bins ${dir}/*)
# Find the latest timestamp of the binaries
set(bin_time 0)
foreach(bin ${bins})
    file(TIMESTAMP ${bin} time)
    if(${time} GREATER ${bin_time})
        set(bin_time ${time})
    endif()
endforeach()
# Find the timestamp of the output file
file(TIMESTAMP ${out} out_time)
# Generate only if one of the binaries is newer than the output file
if(${bin_time} GREATER ${out_time})
    file(WRITE ${out} "")
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
endif()
