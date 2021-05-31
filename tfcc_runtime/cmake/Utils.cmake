function(protobuf_generate_cpp SRCS_REF HDRS_REF BASE_PATH PROTOBUF_PATH)
    get_filename_component(PROTOBUF_PATH ${PROTOBUF_PATH} ABSOLUTE)
    get_filename_component(BASE_PATH ${BASE_PATH} ABSOLUTE)
    file(GLOB_RECURSE PROTOBUF_FILES ${PROTOBUF_PATH} *.proto)
    foreach(PROTOBUF_FILE ${PROTOBUF_FILES})
        file(RELATIVE_PATH RELATIVE_PROTOBUF_FILE ${BASE_PATH} ${PROTOBUF_FILE})
        string(REPLACE .proto .pb.cc SRC ${RELATIVE_PROTOBUF_FILE})
        string(REPLACE .proto .pb.h HDR ${RELATIVE_PROTOBUF_FILE})
        set(HDRS ${HDRS} ${CMAKE_CURRENT_BINARY_DIR}/protos/${HDR})
        set(SRCS ${SRCS} ${CMAKE_CURRENT_BINARY_DIR}/protos/${SRC})
    endforeach()
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/protos
        COMMAND mkdir
        ARGS -p ${CMAKE_CURRENT_BINARY_DIR}/protos
    )
    add_custom_command(
        OUTPUT ${HDRS} ${SRCS}
        COMMAND protobuf::protoc
        ARGS --proto_path ${BASE_PATH} --cpp_out ${CMAKE_CURRENT_BINARY_DIR}/protos ${PROTOBUF_FILES}
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/protos ${PROTOBUF_FILES} protobuf::protoc
        COMMENT "Running protoc"
    )
    set(${HDRS_REF} ${HDRS} PARENT_SCOPE)
    set(${SRCS_REF} ${SRCS} PARENT_SCOPE)
endfunction()

function (search_srcs R_LIST_NAME)
    set(SRCS_SEARCH_PATH ${ARGV})
    list(REMOVE_AT SRCS_SEARCH_PATH 0)
    foreach (DIR ${SRCS_SEARCH_PATH})
        aux_source_directory(${DIR} TMP_SRC_FILES)
        set(DIR_ALL_FILES "")
        FILE(GLOB DIR_ALL_FILES ${DIR}/*)
        source_group(${DIR} ${DIR_ALL_FILES})
    endforeach()
    set(${R_LIST_NAME} ${${R_LIST_NAME}} ${TMP_SRC_FILES} PARENT_SCOPE)
endfunction()
