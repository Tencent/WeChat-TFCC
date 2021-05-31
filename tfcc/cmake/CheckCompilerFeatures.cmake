include(CheckCXXSourceCompiles)

function(is_support_parameter_pack_lambda_capture RESULT)
    unset(IS_SUPPORT_PPLC CACHE)
    check_cxx_source_compiles(
        "template <class... Args>
        void test(Args... args) {
            auto x = [args...]() {};
            x();
        }
        int main() {
            test(1, 2, 3);
            return 0;
        }"
        IS_SUPPORT_PPLC
    )
    set(${RESULT} ${IS_SUPPORT_PPLC} PARENT_SCOPE)
endfunction()
