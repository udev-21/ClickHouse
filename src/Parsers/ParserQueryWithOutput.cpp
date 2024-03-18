#include <Parsers/ASTExplainQuery.h>
#include <Parsers/ASTSelectWithUnionQuery.h>
#include <Parsers/ASTSetQuery.h>
#include <Parsers/ParserAlterQuery.h>
#include <Parsers/ParserBackupQuery.h>
#include <Parsers/ParserCheckQuery.h>
#include <Parsers/ParserCreateQuery.h>
#include <Parsers/ParserDescribeTableQuery.h>
#include <Parsers/ParserDropQuery.h>
#include <Parsers/ParserUndropQuery.h>
#include <Parsers/ParserExplainQuery.h>
#include <Parsers/ParserKillQueryQuery.h>
#include <Parsers/ParserOptimizeQuery.h>
#include <Parsers/ParserQueryWithOutput.h>
#include <Parsers/ParserRenameQuery.h>
#include <Parsers/ParserSelectWithUnionQuery.h>
#include <Parsers/ParserSetQuery.h>
#include <Parsers/ParserShowProcesslistQuery.h>
#include <Parsers/ParserShowTablesQuery.h>
#include <Parsers/ParserShowColumnsQuery.h>
#include <Parsers/ParserShowEngineQuery.h>
#include <Parsers/ParserShowFunctionsQuery.h>
#include <Parsers/ParserShowIndexesQuery.h>
#include <Parsers/ParserShowSettingQuery.h>
#include <Parsers/ParserTablePropertiesQuery.h>
#include <Parsers/ParserWatchQuery.h>
#include <Parsers/ParserDescribeCacheQuery.h>
#include <Parsers/QueryWithOutputSettingsPushDownVisitor.h>
#include <Parsers/Access/ParserShowAccessEntitiesQuery.h>
#include <Parsers/Access/ParserShowAccessQuery.h>
#include <Parsers/Access/ParserShowCreateAccessEntityQuery.h>
#include <Parsers/Access/ParserShowGrantsQuery.h>
#include <Parsers/Access/ParserShowPrivilegesQuery.h>
#include <Common/Exception.h>
#include <Common/assert_cast.h>


namespace DB
{

bool ParserQueryWithOutput::parseImpl(Pos & pos, ASTPtr & node, Expected & expected)
{
    ParserShowTablesQuery show_tables_p;
    ParserShowColumnsQuery show_columns_p;
    ParserShowEnginesQuery show_engine_p;
    ParserShowFunctionsQuery show_functions_p;
    ParserShowIndexesQuery show_indexes_p;
    ParserShowSettingQuery show_setting_p;
    ParserSelectWithUnionQuery select_p;
    ParserTablePropertiesQuery table_p;
    ParserDescribeTableQuery describe_table_p;
    ParserDescribeCacheQuery describe_cache_p;
    ParserShowProcesslistQuery show_processlist_p;
    ParserCreateQuery create_p;
    ParserAlterQuery alter_p;
    ParserRenameQuery rename_p;
    ParserDropQuery drop_p;
    ParserUndropQuery undrop_p;
    ParserCheckQuery check_p;
    ParserOptimizeQuery optimize_p;
    ParserKillQueryQuery kill_query_p;
    ParserWatchQuery watch_p;
    ParserShowAccessQuery show_access_p;
    ParserShowAccessEntitiesQuery show_access_entities_p;
    ParserShowCreateAccessEntityQuery show_create_access_entity_p;
    ParserShowGrantsQuery show_grants_p;
    ParserShowPrivilegesQuery show_privileges_p;
    ParserExplainQuery explain_p(end, allow_settings_after_format_in_insert);
    ParserBackupQuery backup_p;

    ASTPtr query;

    bool parsed =
           explain_p.parse(pos, query, expected)
        || select_p.parse(pos, query, expected)
        || show_create_access_entity_p.parse(pos, query, expected) /// should be before `show_tables_p`
        || show_tables_p.parse(pos, query, expected)
        || show_columns_p.parse(pos, query, expected)
        || show_engine_p.parse(pos, query, expected)
        || show_functions_p.parse(pos, query, expected)
        || show_indexes_p.parse(pos, query, expected)
        || show_setting_p.parse(pos, query, expected)
        || table_p.parse(pos, query, expected)
        || describe_cache_p.parse(pos, query, expected)
        || describe_table_p.parse(pos, query, expected)
        || show_processlist_p.parse(pos, query, expected)
        || create_p.parse(pos, query, expected)
        || alter_p.parse(pos, query, expected)
        || rename_p.parse(pos, query, expected)
        || drop_p.parse(pos, query, expected)
        || undrop_p.parse(pos, query, expected)
        || check_p.parse(pos, query, expected)
        || kill_query_p.parse(pos, query, expected)
        || optimize_p.parse(pos, query, expected)
        || watch_p.parse(pos, query, expected)
        || show_access_p.parse(pos, query, expected)
        || show_access_entities_p.parse(pos, query, expected)
        || show_grants_p.parse(pos, query, expected)
        || show_privileges_p.parse(pos, query, expected)
        || backup_p.parse(pos, query, expected);

    if (!parsed)
        return false;

    auto * query_with_output = typeid_cast<ASTQueryWithOutput *>(query.get());
    chassert(query_with_output, "AST is not a ASTQueryWithOutput");

    ParserKeyword s_into_outfile(Keyword::INTO_OUTFILE);
    if (s_into_outfile.ignore(pos, expected))
    {
        ParserStringLiteral out_file_p;
        auto & out_file_node = ASTHelpers::getOrCreate(query_with_output, query_with_output->out_file);
        if (!out_file_p.parse(pos, out_file_node, expected))
            return false;

        ParserKeyword s_append(Keyword::APPEND);
        if (s_append.ignore(pos, expected))
        {
            query_with_output->is_outfile_append = true;
        }

        ParserKeyword s_truncate(Keyword::TRUNCATE);
        if (s_truncate.ignore(pos, expected))
        {
            query_with_output->is_outfile_truncate = true;
        }

        ParserKeyword s_stdout(Keyword::AND_STDOUT);
        if (s_stdout.ignore(pos, expected))
        {
            query_with_output->is_into_outfile_with_stdout = true;
        }

        ParserKeyword s_compression_method(Keyword::COMPRESSION);
        if (s_compression_method.ignore(pos, expected))
        {
            ParserStringLiteral compression_p;
            auto & compression_node = ASTHelpers::getOrCreate(query_with_output, query_with_output->compression);
            if (!compression_p.parse(pos, compression_node, expected))
                return false;

            ParserKeyword s_compression_level(Keyword::LEVEL);
            if (s_compression_level.ignore(pos, expected))
            {
                ParserNumber compression_level_p;
                auto & compression_level_node = ASTHelpers::getOrCreate(query_with_output, query_with_output->compression_level);
                if (!compression_level_p.parse(pos, compression_level_node, expected))
                    return false;
            }
        }
    }

    ParserKeyword s_format(Keyword::FORMAT);

    if (s_format.ignore(pos, expected))
    {
        ParserIdentifier format_p;
        auto & format_node = ASTHelpers::getOrCreate(query_with_output, query_with_output->format);
        if (!format_p.parse(pos, format_node, expected))
            return false;
        setIdentifierSpecial(format_node);
    }

    // SETTINGS key1 = value1, key2 = value2, ...
    ParserKeyword s_settings(Keyword::SETTINGS);
    if (!query_with_output->settings_ast && s_settings.ignore(pos, expected))
    {
        ParserSetQuery parser_settings(true);
        auto & settings_node = ASTHelpers::getOrCreate(query_with_output, query_with_output->settings_ast);
        if (!parser_settings.parse(pos, settings_node, expected))
            return false;

        // SETTINGS after FORMAT is not parsed by the SELECT parser (ParserSelectQuery)
        // Pass them manually, to apply in InterpreterSelectQuery::initSettings()
        if (query->as<ASTSelectWithUnionQuery>())
        {
            auto settings = query_with_output->settings_ast->clone();
            assert_cast<ASTSetQuery *>(settings.get())->print_in_format = false;
            QueryWithOutputSettingsPushDownVisitor::Data data{settings};
            QueryWithOutputSettingsPushDownVisitor(data).visit(query);
        }
    }

    node = std::move(query);
    return true;
}

}
