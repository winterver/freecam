# edited from https://stackoverflow.com/questions/11813271/embed-resources-eg-shader-code-images-into-executable-library-with-cmake

# Collect input files
file(GLOB bins ${dir}/*)
# Find the latest timestamp of the input files
set(bin_time 0)
foreach(bin ${bins})
    file(TIMESTAMP ${bin} time "%y%m%d%H%M%S")
    if(${time} GREATER ${bin_time})
        set(bin_time ${time})
    endif()
endforeach()
# Find the timestamp of the output file
file(TIMESTAMP ${out} out_time "%y%m%d%H%M%S")
# Set out_time to 0 if file ${out} doesn't exist yet
if(NOT EXISTS ${out})
    set(out_time 0)
endif()
# Generate only if one of the input files is newer than the output file
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
        # Prepare regex
        string(REPEAT "([0-9a-f][0-9a-f])" 4 expr)
        # Convert hex data for C compatibility
        string(REGEX REPLACE ${expr} "0x\\4\\3\\2\\1," filedata ${filedata})
        # Append data to output file
        file(APPEND ${out} "const uint32_t ${filename}_code[] = {${filedata}};\n")
        file(APPEND ${out} "const unsigned ${filename}_size = sizeof(${filename}_code);\n")
    endforeach()
endif()
