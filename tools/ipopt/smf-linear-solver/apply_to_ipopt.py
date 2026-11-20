#!/usr/bin/env python3
"""Apply native linear_solver=smf support to an Ipopt 3.14.x source tree."""

import pathlib
import shutil
import sys


def patch_once(root: pathlib.Path, rel: str, old: str, new: str) -> None:
    p = root / rel
    txt = p.read_text(encoding="utf-8")
    if new in txt:
        return
    if old not in txt:
        raise RuntimeError(f"Expected anchor not found in {rel}:\n{old[:120]}")
    p.write_text(txt.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/Ipopt-3.14.x", file=sys.stderr)
        sys.exit(2)

    root = pathlib.Path(sys.argv[1]).resolve()
    src_dir = pathlib.Path(__file__).resolve().parent

    shutil.copy2(
        src_dir / "IpSmfSolverInterface.hpp",
        root / "src/Algorithm/LinearSolvers/IpSmfSolverInterface.hpp",
    )
    shutil.copy2(
        src_dir / "IpSmfSolverInterface.cpp",
        root / "src/Algorithm/LinearSolvers/IpSmfSolverInterface.cpp",
    )

    # 1. IpLinearSolvers.h
    patch_once(
        root,
        "src/Algorithm/LinearSolvers/IpLinearSolvers.h",
        "#define IPOPTLINEARSOLVER_MUMPS   0x400u",
        "#define IPOPTLINEARSOLVER_MUMPS   0x400u\n#define IPOPTLINEARSOLVER_SMF     0x800u",
    )

    # 2. IpLinearSolvers.c
    patch_once(
        root,
        "src/Algorithm/LinearSolvers/IpLinearSolvers.c",
        "#ifdef IPOPT_HAS_MUMPS\n   solvers |= IPOPTLINEARSOLVER_MUMPS;\n#endif",
        "#ifdef IPOPT_HAS_MUMPS\n   solvers |= IPOPTLINEARSOLVER_MUMPS;\n#endif\n\n"
        "#if !defined(IPOPT_SINGLE) && !defined(IPOPT_INT64) && defined(IPOPT_HAS_SMF)\n"
        "   solvers |= IPOPTLINEARSOLVER_SMF;\n#endif",
    )
    patch_once(
        root,
        "src/Algorithm/LinearSolvers/IpLinearSolvers.c",
        "      solvers |= IPOPTLINEARSOLVER_PARDISO;\n#endif",
        "      solvers |= IPOPTLINEARSOLVER_PARDISO;\n"
        "#ifndef IPOPT_SINGLE\n      solvers |= IPOPTLINEARSOLVER_SMF;\n#endif\n#endif",
    )

    # 3. IpLinearSolversRegOp.cpp
    patch_once(
        root,
        "src/Algorithm/LinearSolvers/IpLinearSolversRegOp.cpp",
        '#include "IpPardisoSolverInterface.hpp"',
        '#include "IpPardisoSolverInterface.hpp"\n#ifndef IPOPT_SINGLE\n#include "IpSmfSolverInterface.hpp"\n#endif',
    )
    patch_once(
        root,
        "src/Algorithm/LinearSolvers/IpLinearSolversRegOp.cpp",
        '   if( availablesolvers & IPOPTLINEARSOLVER_PARDISO )\n   {\n'
        '      roptions->SetRegisteringCategory("Pardiso (pardiso-project.org) Linear Solver");\n'
        '      PardisoSolverInterface::RegisterOptions(roptions);\n   }\n#endif',
        '   if( availablesolvers & IPOPTLINEARSOLVER_PARDISO )\n   {\n'
        '      roptions->SetRegisteringCategory("Pardiso (pardiso-project.org) Linear Solver");\n'
        '      PardisoSolverInterface::RegisterOptions(roptions);\n   }\n'
        '#ifndef IPOPT_SINGLE\n'
        '   if( availablesolvers & IPOPTLINEARSOLVER_SMF )\n   {\n'
        '      roptions->SetRegisteringCategory("SMF Linear Solver");\n'
        '      SmfSolverInterface::RegisterOptions(roptions);\n   }\n'
        '#endif\n#endif',
    )

    # 4. IpAlgBuilder.hpp
    patch_once(
        root,
        "src/Algorithm/IpAlgBuilder.hpp",
        "   SmartPtr<LibraryLoader> GetPardisoLoader(\n"
        "      const OptionsList& options,\n"
        "      const std::string& prefix\n"
        "   );",
        "   SmartPtr<LibraryLoader> GetPardisoLoader(\n"
        "      const OptionsList& options,\n"
        "      const std::string& prefix\n"
        "   );\n\n"
        "   /// Gives Library Loader for smf shared C ABI library (libsmf_c).\n"
        "   SmartPtr<LibraryLoader> GetSmfLoader(\n"
        "      const OptionsList& options,\n"
        "      const std::string& prefix\n"
        "   );",
    )
    patch_once(
        root,
        "src/Algorithm/IpAlgBuilder.hpp",
        "   /// loader of Pardiso library (at runtime)\n   SmartPtr<LibraryLoader> pardisoloader;",
        "   /// loader of Pardiso library (at runtime)\n   SmartPtr<LibraryLoader> pardisoloader;\n"
        "   /// loader of smf library (at runtime)\n   SmartPtr<LibraryLoader> smfloader;",
    )

    # 5. IpAlgBuilder.cpp
    patch_once(
        root,
        "src/Algorithm/IpAlgBuilder.cpp",
        '#include "IpPardisoSolverInterface.hpp"',
        '#include "IpPardisoSolverInterface.hpp"\n#ifndef IPOPT_SINGLE\n#include "IpSmfSolverInterface.hpp"\n#endif',
    )
    patch_once(
        root,
        "src/Algorithm/IpAlgBuilder.cpp",
        '   if( availablesolvers & IPOPTLINEARSOLVER_MUMPS )\n   {\n'
        '      options.push_back("mumps");\n'
        '      descrs.push_back("use the Mumps package");\n   }',
        '   if( availablesolvers & IPOPTLINEARSOLVER_SMF )\n   {\n'
        '      options.push_back("smf");\n'
        '      if( availablesolverslinked & IPOPTLINEARSOLVER_SMF )\n      {\n'
        '         descrs.push_back("use the smf sparse symmetric multifrontal solver");\n'
        '      }\n      else\n      {\n'
        '         descrs.push_back("load the smf sparse symmetric multifrontal solver from user-provided library at runtime");\n'
        '      }\n   }\n\n'
        '   if( availablesolvers & IPOPTLINEARSOLVER_MUMPS )\n   {\n'
        '      options.push_back("mumps");\n'
        '      descrs.push_back("use the Mumps package");\n   }',
    )
    patch_once(
        root,
        "src/Algorithm/IpAlgBuilder.cpp",
        '   else if( availablesolverslinked & IPOPTLINEARSOLVER_MUMPS )\n   {\n'
        '      defaultsolver = "mumps";\n   }',
        '   else if( availablesolverslinked & IPOPTLINEARSOLVER_SMF )\n   {\n'
        '      defaultsolver = "smf";\n   }\n'
        '   else if( availablesolverslinked & IPOPTLINEARSOLVER_MUMPS )\n   {\n'
        '      defaultsolver = "mumps";\n   }',
    )
    patch_once(
        root,
        "src/Algorithm/IpAlgBuilder.cpp",
        '   roptions->AddStringOption1(\n'
        '      "pardisolib", "Name of library containing Pardiso routines (from pardiso-project.org) for load at runtime",\n'
        '#ifdef PARDISO_LIB\n      PARDISO_LIB,\n#else\n'
        '      "libpardiso." IPOPT_SHAREDLIBEXT,\n#endif\n'
        '      "*", "Any acceptable filename (may contain path, too)");',
        '   roptions->AddStringOption1(\n'
        '      "pardisolib", "Name of library containing Pardiso routines (from pardiso-project.org) for load at runtime",\n'
        '#ifdef PARDISO_LIB\n      PARDISO_LIB,\n#else\n'
        '      "libpardiso." IPOPT_SHAREDLIBEXT,\n#endif\n'
        '      "*", "Any acceptable filename (may contain path, too)");\n\n'
        '   if( (availablesolverslinked ^ availablesolvers) & IPOPTLINEARSOLVER_SMF )\n'
        '      roptions->AddStringOption1(\n'
        '         "smflib", "Name of library containing smf C ABI routines (libsmf_c) for load at runtime",\n'
        '#if defined(_WIN32) && defined(_MSC_VER)\n         "smf_c.dll",\n#else\n'
        '         "libsmf_c." IPOPT_SHAREDLIBEXT,\n#endif\n'
        '         "*", "Any acceptable filename (may contain path, too)");',
    )
    patch_once(
        root,
        "src/Algorithm/IpAlgBuilder.cpp",
        '   else if( linear_solver == "pardiso" )\n   {\n'
        '      SolverInterface = new PardisoSolverInterface(GetPardisoLoader(options, prefix));\n   }\n#endif',
        '   else if( linear_solver == "pardiso" )\n   {\n'
        '      SolverInterface = new PardisoSolverInterface(GetPardisoLoader(options, prefix));\n   }\n'
        '#ifndef IPOPT_SINGLE\n'
        '   else if( linear_solver == "smf" )\n   {\n'
        '      SolverInterface = new SmfSolverInterface(\n'
        '#ifdef IPOPT_HAS_SMF\n         NULL\n#else\n         GetSmfLoader(options, prefix)\n#endif\n'
        '      );\n'
        '      linear_solver = SmfSolverInterface::GetName();\n'
        '   }\n#endif\n#endif',
    )
    patch_once(
        root,
        "src/Algorithm/IpAlgBuilder.cpp",
        "   return pardisoloader;\n}\n\n} // namespace",
        "   return pardisoloader;\n}\n\n"
        "SmartPtr<LibraryLoader> AlgorithmBuilder::GetSmfLoader(\n"
        "   const OptionsList& options,\n"
        "   const std::string& prefix\n)\n{\n"
        "   if( !IsValid(smfloader) )\n   {\n"
        '      std::string libname;\n'
        '      options.GetStringValue("smflib", libname, prefix);\n'
        "      smfloader = new LibraryLoader(libname);\n   }\n\n"
        "   return smfloader;\n}\n\n} // namespace",
    )

    # 6. configure.ac
    patch_once(
        root,
        "configure.ac",
        "#########\n# SPRAL #\n#########",
        "#######\n# SMF #\n#######\n\n"
        "if test $precision$intsize = double32 ; then\n"
        "  AC_COIN_CHK_PKG(Smf,[IpoptLib],[smf])\n"
        "else\n"
        "  AM_CONDITIONAL([COIN_HAS_SMF],[false])\n"
        "fi\n\n"
        "#########\n# SPRAL #\n#########",
    )

    # 7. src/Makefile.am
    patch_once(
        root,
        "src/Makefile.am",
        "  Algorithm/LinearSolvers/IpPardisoSolverInterface.cpp",
        "  Algorithm/LinearSolvers/IpPardisoSolverInterface.cpp \\\n"
        "  Algorithm/LinearSolvers/IpSmfSolverInterface.cpp",
    )

    # 8. src/Makefile.in (when present in release tarballs)
    if (root / "src/Makefile.in").exists():
        patch_once(
            root,
            "src/Makefile.in",
            "  Algorithm/LinearSolvers/IpPardisoSolverInterface.cpp",
            "  Algorithm/LinearSolvers/IpPardisoSolverInterface.cpp \\\n"
            "  Algorithm/LinearSolvers/IpSmfSolverInterface.cpp",
        )
        patch_once(
            root,
            "src/Makefile.in",
            "\tAlgorithm/LinearSolvers/IpPardisoSolverInterface.lo",
            "\tAlgorithm/LinearSolvers/IpPardisoSolverInterface.lo \\\n"
            "\tAlgorithm/LinearSolvers/IpSmfSolverInterface.lo",
        )
        patch_once(
            root,
            "src/Makefile.in",
            "\tAlgorithm/LinearSolvers/$(DEPDIR)/IpPardisoSolverInterface.Plo \\",
            "\tAlgorithm/LinearSolvers/$(DEPDIR)/IpPardisoSolverInterface.Plo \\\n"
            "\tAlgorithm/LinearSolvers/$(DEPDIR)/IpSmfSolverInterface.Plo \\",
        )
        patch_once(
            root,
            "src/Makefile.in",
            "Algorithm/LinearSolvers/IpPardisoSolverInterface.lo:  \\\n"
            "\tAlgorithm/LinearSolvers/$(am__dirstamp) \\\n"
            "\tAlgorithm/LinearSolvers/$(DEPDIR)/$(am__dirstamp)",
            "Algorithm/LinearSolvers/IpPardisoSolverInterface.lo:  \\\n"
            "\tAlgorithm/LinearSolvers/$(am__dirstamp) \\\n"
            "\tAlgorithm/LinearSolvers/$(DEPDIR)/$(am__dirstamp)\n"
            "Algorithm/LinearSolvers/IpSmfSolverInterface.lo:  \\\n"
            "\tAlgorithm/LinearSolvers/$(am__dirstamp) \\\n"
            "\tAlgorithm/LinearSolvers/$(DEPDIR)/$(am__dirstamp)",
        )
        patch_once(
            root,
            "src/Makefile.in",
            "@AMDEP_TRUE@@am__include@ @am__quote@Algorithm/LinearSolvers/$(DEPDIR)/IpPardisoSolverInterface.Plo@am__quote@ # am--include-marker",
            "@AMDEP_TRUE@@am__include@ @am__quote@Algorithm/LinearSolvers/$(DEPDIR)/IpPardisoSolverInterface.Plo@am__quote@ # am--include-marker\n"
            "@AMDEP_TRUE@@am__include@ @am__quote@Algorithm/LinearSolvers/$(DEPDIR)/IpSmfSolverInterface.Plo@am__quote@ # am--include-marker",
        )
        p = root / "src/Makefile.in"
        txt = p.read_text(encoding="utf-8")
        rm_old = "\t-rm -f Algorithm/LinearSolvers/$(DEPDIR)/IpPardisoSolverInterface.Plo\n"
        rm_new = (
            "\t-rm -f Algorithm/LinearSolvers/$(DEPDIR)/IpPardisoSolverInterface.Plo\n"
            "\t-rm -f Algorithm/LinearSolvers/$(DEPDIR)/IpSmfSolverInterface.Plo\n"
        )
        if rm_new not in txt:
            p.write_text(txt.replace(rm_old, rm_new), encoding="utf-8")

    print(f"Applied native linear_solver=smf support to {root}")


if __name__ == "__main__":
    main()
