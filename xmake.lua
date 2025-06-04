add_rules("plugin.compile_commands.autoupdate")
add_rules("mode.release", "mode.debug")

add_requires("nlohmann_json")

target("tiny-llm")
	set_kind("binary")
    add_packages("nlohmann_json")
	set_languages("c++23")
	add_files("src/**.cpp")
