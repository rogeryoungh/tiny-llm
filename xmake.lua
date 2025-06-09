add_rules("plugin.compile_commands.autoupdate")
add_rules("mode.release", "mode.debug")

option("use_blas")
	set_showmenu(true)
    set_description("Enable BLAS support")
option_end()


add_requires("nlohmann_json")

if has_config("use_blas") then
	add_requires("openblas")
end

target("tiny-llm")
	set_kind("binary")
    add_packages("nlohmann_json")
	if has_config("use_blas") then
		add_packages("openblas")
		add_defines("TINYLLM_USE_BLAS")
	end
	add_vectorexts("avx", "avx2", "avx512")
	set_languages("c++23")
	add_files("src/**.cpp")
	-- set_policy("build.sanitizer.address", true)

